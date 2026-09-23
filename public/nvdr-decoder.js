/*
 * Decoding off the page's thread. A 13.5 Mpx photo takes seconds to
 * decode in JS, and on the main thread that is seconds of a page that
 * neither scrolls nor answers a click. Here the same work (nvdr-tasks.js)
 * runs in module workers, and the pixels come back transferred, not
 * copied:
 *
 *   import { decoderFor } from './nvdr-decoder.js';
 *   const d = decoderFor('page');
 *   await d.run({ op: 'put', key: 'a', bytes });
 *   const r = await d.run({ op: 'decode', key: 'a', length: n });
 *
 * A key always goes to the same worker, so whatever a worker keeps for it
 * (a stored container, an open album, the ranges fetched from an album
 * URL) is there on the next call. A browser that cannot start a module
 * worker gets the same answers from the main thread: the tasks that were
 * waiting, and the containers that had been put, are replayed in place.
 */
import { runTask } from './nvdr-tasks.js';

class Decoder {
    constructor() {
        this.pending = new Map();   // id -> { task, resolve, reject }
        this.puts = new Map();      // key -> task, to replay if the worker dies
        this.next = 1;
        this.worker = null;
        try {
            this.worker = new Worker(new URL('./nvdr-worker.js', import.meta.url), { type: 'module' });
            this.worker.onmessage = ({ data }) => this.settle(data);
            this.worker.onerror = (e) => { e.preventDefault?.(); this.fallBack(); };
        } catch {
            this.worker = null;
        }
    }

    /* Whether tasks run in a worker (false once fallen back). */
    get offThread() { return !!this.worker; }

    /* Runs a task; resolves with its result, rejects with its error. */
    run(task) {
        if (task.op === 'put') this.puts.set(task.key, task);
        if (task.op === 'drop') this.puts.delete(task.key);
        if (!this.worker) return this.inline(task);
        return new Promise((resolve, reject) => {
            const id = this.next++;
            this.pending.set(id, { task, resolve, reject });
            this.worker.postMessage({ ...task, id });
        });
    }

    async inline(task) {
        const { result } = await runTask(task);
        return result;
    }

    settle({ id, result, error }) {
        const p = this.pending.get(id);
        if (!p) return;
        this.pending.delete(id);
        if (error !== undefined) p.reject(new Error(error));
        else p.resolve(result);
    }

    /* The worker could not load or died: from here on, the main thread. */
    fallBack() {
        if (!this.worker) return;
        this.worker.terminate();
        this.worker = null;
        const waiting = [...this.pending.values()];
        this.pending.clear();
        const replayed = new Set(waiting.filter(p => p.task.op === 'put').map(p => p.task.key));
        const restore = [...this.puts.values()].filter(t => !replayed.has(t.key));
        (async () => {
            for (const t of restore) await this.inline(t);
            for (const p of waiting) this.inline(p.task).then(p.resolve, p.reject);
        })();
    }
}

const pool = [];
const size = Math.max(1, Math.min(4, ((typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2) - 1));

/* The decoder that serves `key`: one of a few workers, always the same
 * one for the same key. */
export function decoderFor(key = '') {
    let h = 0;
    for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
    const k = Math.abs(h) % size;
    if (!pool[k]) pool[k] = new Decoder();
    return pool[k];
}
