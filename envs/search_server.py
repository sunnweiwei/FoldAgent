#!/usr/bin/env python3
"""
python search_server.py --host 0.0.0.0 --port 8000
"""

import os
import json
import time
import pickle
import re
import logging
import argparse
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from queue import Empty, Full
import signal
import sys
import threading
from collections import deque, defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Minimal logging for performance
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.disabled = True

MODEL_NAME = 'Qwen/Qwen3-Embedding-8B'
CORPUS_DATASET = "Tevatron/browsecomp-plus-corpus"
CORPUS_EMBEDDING_DATASET = "miaolu3/browsecomp-plus"
CORPUS_EMBEDDING_FILE = "corpus_embeddings.pkl"

@dataclass
class SearchRequest:
    query: str
    k: int = 20
    request_id: str = None


@dataclass
class SearchBatch:
    requests: List[SearchRequest]


class QueryRequest(BaseModel):
    query: str
    k: int = 20

class OpenRequest(BaseModel):
    docid: Optional[str] = None
    url: Optional[str] = None



class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    took_ms: float


class FastMetricsTracker:
    """Lightweight metrics tracker"""

    def __init__(self):
        self.request_count = 0
        self.total_time = 0.0
        self.max_time = 0.0
        self.last_reset = time.time()
        self.lock = threading.Lock()

    def record_request(self, process_time_ms: float):
        with self.lock:
            self.request_count += 1
            self.total_time += process_time_ms
            self.max_time = max(self.max_time, process_time_ms)

    def get_stats_and_reset(self):
        with self.lock:
            if self.request_count == 0:
                return 0, 0.0, 0.0

            avg_time = self.total_time / self.request_count
            stats = (self.request_count, avg_time, self.max_time)

            # Reset
            self.request_count = 0
            self.total_time = 0.0
            self.max_time = 0.0
            self.last_reset = time.time()

            return stats


def last_token_pool(last_hidden_states, attention_mask):
    """Pool embeddings using last token"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def keep_first_n_words(text: str, n: int = 1000) -> str:
    if not text:
        return ""
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == n:
            return text[:m.end()] + '\n[Document is truncated.]'
    return text


def load_corpus():
    """Load the corpus dataset from HuggingFace"""
    print(f"Loading corpus dataset from {CORPUS_DATASET}...")
    ds = load_dataset(CORPUS_DATASET, split='train')
    docid_to_text = {row["docid"]: {
        'raw': keep_first_n_words(row["text"], 15000),
        'content': keep_first_n_words(row["text"], 1000),
        'url': row['url'],
        'docid': row['docid']
    } for row in ds}
    url_to_docid = {row["url"]: row['docid'] for row in ds}
    print(f"Loaded {len(docid_to_text)} documents")
    return docid_to_text, url_to_docid


def encode_corpus():
    """Load corpus embeddings from HuggingFace dataset"""
    from huggingface_hub import hf_hub_download

    print(f"Downloading corpus embeddings from {CORPUS_EMBEDDING_DATASET}...")
    embeddings_path = hf_hub_download(
        repo_id=CORPUS_EMBEDDING_DATASET,
        filename=CORPUS_EMBEDDING_FILE,
        repo_type="dataset"
    )
    print(f"Loading corpus embeddings from {embeddings_path}...")
    with open(embeddings_path, 'rb') as f:
        return pickle.load(f)


def high_speed_batcher(request_queue: mp.Queue, batch_queue: mp.Queue,
                       max_batch_size: int = 512, batch_timeout: float = 0.005):
    """Ultra-fast batcher optimized for high throughput"""

    batch_requests = []
    last_batch_time = time.time()

    while True:
        try:
            # Very short timeout for maximum responsiveness
            request = request_queue.get(timeout=batch_timeout)

            if request is None:  # Shutdown
                if batch_requests:
                    batch_queue.put(SearchBatch(requests=batch_requests))
                break

            batch_requests.append(request)
            current_time = time.time()

            # Send batch if full or timeout exceeded
            if (len(batch_requests) >= max_batch_size or
                    (current_time - last_batch_time) >= batch_timeout):
                batch_queue.put_nowait(SearchBatch(requests=batch_requests))
                batch_requests = []
                last_batch_time = current_time

        except Empty:
            # Process any pending requests immediately for low latency
            if batch_requests:
                current_time = time.time()
                if (current_time - last_batch_time) >= batch_timeout:
                    batch_queue.put_nowait(SearchBatch(requests=batch_requests))
                    batch_requests = []
                    last_batch_time = current_time
            continue

        except Full:
            # If batch queue is full, just reset and continue
            batch_requests = []
            last_batch_time = time.time()
            continue

        except Exception:
            # On any error, reset batch to avoid getting stuck
            batch_requests = []
            last_batch_time = time.time()


def optimized_worker(gpu_id: int, batch_queue: mp.Queue, result_queue: mp.Queue,
                     corpus_data: Dict):
    """Optimized worker for maximum throughput"""
    try:
        # Set GPU device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device("cuda:0")

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                          attn_implementation="flash_attention_2").to(device)
        model.eval()

        # Load corpus embeddings
        corpus_embeddings = corpus_data['embeddings'].to(device)
        corpus_docids = corpus_data['docids']
        task_description = 'Given a web search query, retrieve relevant passages that answer the query'

        print(f"Worker {gpu_id}: Ready")

        while True:
            try:
                # Very short timeout for responsiveness
                batch = batch_queue.get(timeout=0.01)

                if batch is None:  # Shutdown
                    break

                # Process batch immediately
                fast_process_batch(batch, tokenizer, model, device,
                                   corpus_embeddings, corpus_docids, task_description, result_queue)

            except Empty:
                continue
            except Exception as e:
                # Log critical errors only
                print(f"Worker {gpu_id} error: {e}")

    except Exception as e:
        print(f"Worker {gpu_id} init error: {e}")


def fast_process_batch(batch: SearchBatch, tokenizer, model, device,
                       corpus_embeddings, corpus_docids, task_description, result_queue):
    """Ultra-fast batch processing"""
    try:
        # Prepare queries
        queries = [get_detailed_instruct(task_description, req.query) for req in batch.requests]

        # Tokenize
        batch_dict = tokenizer(queries, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        # Encode
        with torch.no_grad():
            outputs = model(**batch_dict)
            query_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            similarities = torch.mm(query_embeddings, corpus_embeddings.T)

        # Process results
        for i, request in enumerate(batch.requests):
            scores, indices = torch.topk(similarities[i], k=min(request.k, len(corpus_docids)))

            results = []
            for score, idx in zip(scores.cpu().tolist(), indices.cpu().tolist()):
                results.append({'docid': corpus_docids[idx], 'score': float(score)})

            result_queue.put({
                'request_id': request.request_id,
                'results': results,
                'status': 'success'
            })

    except Exception as e:
        # Send error for all requests
        for request in batch.requests:
            result_queue.put({
                'request_id': request.request_id,
                'results': [],
                'status': 'error',
                'error': str(e)
            })


class HighThroughputSearchServer:
    def __init__(self, num_gpus: int = 8, max_batch_size: int = 512, batch_timeout: float = 0.005):

        # Optimized for 10k+ requests: Large queues, fast timeouts
        self.request_queue = mp.Queue(maxsize=20000)  # Large queue for burst traffic
        self.batch_queue = mp.Queue(maxsize=1000)  # Large batch queue
        self.result_queue = mp.Queue(maxsize=20000)  # Large result queue

        self.pending_requests = {}
        self.metrics = FastMetricsTracker()

        # Load corpus
        self.docid_to_text, self.url_to_docid = load_corpus()

        # Load embeddings
        self.corpus_data = encode_corpus()

        # Start high-speed batcher
        self.batcher = mp.Process(target=high_speed_batcher,
                                  args=(self.request_queue, self.batch_queue, max_batch_size, batch_timeout))
        self.batcher.start()

        # Start optimized workers
        self.workers = []
        for gpu_id in range(num_gpus):
            worker = mp.Process(target=optimized_worker,
                                args=(gpu_id, self.batch_queue, self.result_queue, self.corpus_data))
            worker.start()
            self.workers.append(worker)

        print(f"🚀 Started {num_gpus} workers with max_batch_size={max_batch_size}")

        # Fast result collector
        self.result_thread = threading.Thread(target=self._fast_collect_results, daemon=True)
        self.result_thread.start()

        # Simple metrics logger
        self.metrics_thread = threading.Thread(target=self._simple_metrics, daemon=True)
        self.metrics_thread.start()

    def _fast_collect_results(self):
        """Optimized result collection"""
        while True:
            try:
                result = self.result_queue.get(timeout=0.1)
                request_id = result['request_id']

                if request_id in self.pending_requests:
                    future = self.pending_requests.pop(request_id)
                    future.set_result(result)

            except Empty:
                continue
            except Exception:
                continue

    def _simple_metrics(self):
        """Simple metrics logging"""
        while True:
            try:
                time.sleep(10.0)
                req_count, avg_time, max_time = self.metrics.get_stats_and_reset()
                if req_count > 0:
                    print(f"📊 10s: {req_count} req | {avg_time:.1f}ms avg | {max_time:.1f}ms max | "
                          f"Queues: {self.request_queue.qsize()}/{self.batch_queue.qsize()}/{self.result_queue.qsize()}")
            except Exception:
                continue

    async def search(self, query: str, k: int = 20) -> Dict:
        """Fast search with minimal overhead"""
        import asyncio
        import uuid

        start_time = time.time()
        request_id = str(uuid.uuid4())
        request = SearchRequest(query=query, k=k, request_id=request_id)

        # Create future
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_requests[request_id] = future

        try:
            # Submit request
            self.request_queue.put_nowait(request)

            # Wait for result with generous timeout for high load
            result = await asyncio.wait_for(future, timeout=60.0)

            # Record metrics
            process_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_request(process_time_ms)

            return result

        except Full:
            self.pending_requests.pop(request_id, None)
            raise HTTPException(status_code=503, detail="Server overloaded")
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise HTTPException(status_code=408, detail="Timeout")

    def shutdown(self):
        """Clean shutdown"""
        print("Shutting down...")

        # Stop batcher
        self.request_queue.put(None)
        self.batcher.join(timeout=2.0)
        if self.batcher.is_alive():
            self.batcher.terminate()

        # Stop workers
        for _ in self.workers:
            self.batch_queue.put(None)

        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()


# Global server
search_server: HighThroughputSearchServer = None

# FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup_event():
    global search_server
    detected_gpus = torch.cuda.device_count()
    num_gpus = int(os.getenv("NUM_GPUS", str(detected_gpus)))
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "2048"))  # Larger default
    batch_timeout = float(os.getenv("BATCH_TIMEOUT", "0.005"))  # 5ms default

    print(f"🚀 Starting search server: {num_gpus} GPUs, batch={max_batch_size}, timeout={batch_timeout}s")

    search_server = HighThroughputSearchServer(
        num_gpus=num_gpus, max_batch_size=max_batch_size, batch_timeout=batch_timeout)


@app.on_event("shutdown")
async def shutdown_event():
    if search_server:
        search_server.shutdown()


@app.post("/search", response_model=QueryResponse)
async def search_endpoint(request: QueryRequest):
    """Optimized search endpoint"""
    start_time = time.time()
    result = await search_server.search(request.query, request.k)

    if result['status'] == 'error':
        raise HTTPException(status_code=500, detail=result.get('error', 'Error'))

    # Add document text
    enriched_results = []
    for res in result['results']:
        text = 'Fetch document error.'
        url = 'ERROR'
        if res['docid'] in search_server.docid_to_text:
            text = search_server.docid_to_text.get(res['docid'])['content']
            url = search_server.docid_to_text.get(res['docid'])['url']
        enriched_results.append({
            'docid': res['docid'],
            'url': url,
            'text':text,
            'score': res['score']
        })

    return QueryResponse(results=enriched_results, took_ms=(time.time() - start_time) * 1000)


@app.post("/open", response_model=QueryResponse)
async def open_page(request: OpenRequest):
    """Optimized search endpoint"""
    start_time = time.time()
    docid = request.docid or (search_server.url_to_docid.get(request.url) if request.url else None)
    if not docid:
        return QueryResponse(
            results=[{'docid': docid, 'url': request.url, 'text': "Missing docid and url, or url not indexed."}],
            took_ms=(time.time() - start_time) * 1000
        )

    item = search_server.docid_to_text.get(docid)
    if not item:
        return QueryResponse(
            results=[{'docid': docid, 'url': request.url, 'text': "Document not found for given docid."}],
            took_ms=(time.time() - start_time) * 1000
        )

    enriched_results = [{
        'docid': docid,
        'url': item.get('url', request.url),
        'text': item.get('raw', item.get('text', "")),
    }]
    return QueryResponse(results=enriched_results, took_ms=(time.time() - start_time) * 1000)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "workers": len(search_server.workers) if search_server else 0}


@app.get("/stats")
async def get_stats():
    if not search_server:
        return {"error": "Not initialized"}

    return {
        "corpus_size": len(search_server.docid_to_text),
        "num_workers": len(search_server.workers),
        "queue_sizes": {
            "requests": search_server.request_queue.qsize(),
            "batches": search_server.batch_queue.qsize(),
            "results": search_server.result_queue.qsize()
        },
        "pending": len(search_server.pending_requests)
    }


def signal_handler(sig, frame):
    if search_server:
        search_server.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=MODEL_NAME)
    parser.add_argument('--corpus', type=str, default=CORPUS_DATASET)
    parser.add_argument('--corpus-embedding-dataset', type=str, default=CORPUS_EMBEDDING_DATASET)
    parser.add_argument('--corpus-embedding-file', type=str, default=CORPUS_EMBEDDING_FILE)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    MODEL_NAME = args.model
    CORPUS_DATASET = args.corpus
    CORPUS_EMBEDDING_DATASET = args.corpus_embedding_dataset
    CORPUS_EMBEDDING_FILE = args.corpus_embedding_file

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)