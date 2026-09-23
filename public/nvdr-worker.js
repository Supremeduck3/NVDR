/*
 * A module worker that decodes off the page's thread (see nvdr-tasks.js
 * for the work and nvdr-decoder.js for the side that talks to it).
 * Messages are { id, ...task }; replies are { id, result } or
 * { id, error }, with the result's pixel buffers transferred.
 */
import { runTask } from './nvdr-tasks.js';

self.onmessage = async ({ data }) => {
    try {
        const { result, transfer } = await runTask(data);
        self.postMessage({ id: data.id, result }, transfer);
    } catch (err) {
        self.postMessage({ id: data.id, error: String(err && err.message || err) });
    }
};
