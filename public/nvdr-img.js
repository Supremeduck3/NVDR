/*
 * <nvdr-img>: NVDR images on any page, the way <img> shows a JPEG.
 *
 *   <script type="module" src="nvdr-img.js"></script>
 *   <nvdr-img src="foto.nvdr" alt="..."></nvdr-img>
 *   <nvdr-img src="galeria.nvda#3" alt="..."></nvdr-img>   third photo of an album
 *   <nvdr-img src="galeria.nvda#praia.jpg"></nvdr-img>       or by the name it was packed with
 *
 * What it keeps from the format, in production:
 *
 * - A single image is painted while it downloads. Every prefix of the file
 *   past its first bytes is a picture, so the element redraws as chunks
 *   arrive: flat colour first, then coarse texture over the whole image,
 *   then detail, instead of a blank box until the last byte.
 * - An album photo costs its own bytes and its references', not the
 *   album's. The index at the front says where everything is, and range
 *   requests fetch the photo and the chain of earlier photos it was
 *   predicted from. A server that ignores ranges sends the whole album,
 *   which still works.
 *
 * `throttle="bytes-per-second"` slows the download down, to watch the
 * progressive render on a fast connection. The element fires `load` with
 * { detail: { bytes } } when it is done and `error` when it cannot be
 * shown.
 */
import { decode, paintFitted } from './nvdr.js';
import { openAlbumImage } from './nvda.js';

class NvdrImg extends HTMLElement {
    static get observedAttributes() { return ['src']; }

    constructor() {
        super();
        const root = this.attachShadow({ mode: 'open' });
        root.innerHTML = `<style>
            :host { display: inline-block; }
            canvas { display: block; width: 100%; height: auto; }
        </style><canvas role="img"></canvas>`;
        this.canvas = root.querySelector('canvas');
        this.generation = 0;
    }

    connectedCallback() { this.load(); }
    attributeChangedCallback() { if (this.isConnected) this.load(); }

    /* At the size the element is shown, averaged down (see paintFitted):
     * a photo shrunk into a thumbnail keeps its grain as grain. */
    paint(rgb, width, height) {
        paintFitted(this.canvas, rgb, width, height);
    }

    async load() {
        const src = this.getAttribute('src');
        if (!src) return;
        const generation = ++this.generation;
        this.canvas.setAttribute('aria-label', this.getAttribute('alt') || '');
        const hash = src.indexOf('#');
        const url = hash >= 0 ? src.slice(0, hash) : src;
        const which = hash >= 0 ? decodeURIComponent(src.slice(hash + 1)) : null;
        try {
            let bytes;
            if (url.toLowerCase().endsWith('.nvda')) {
                // "#3" is the third photo; anything else is a name.
                const key = /^\d+$/.test(which || '') ? Number(which) - 1 : (which || 0);
                const got = await openAlbumImage(url, key);
                if (generation !== this.generation) return;
                if (!got.image) throw new Error('photo did not arrive');
                this.paint(got.image.rgb, got.image.width, got.image.height);
                bytes = got.fetched;
            } else {
                bytes = await this.stream(url, generation);
            }
            if (generation === this.generation)
                this.dispatchEvent(new CustomEvent('load', { detail: { bytes } }));
        } catch (err) {
            if (generation === this.generation)
                this.dispatchEvent(new CustomEvent('error', { detail: { message: err.message } }));
        }
    }

    /* Fetch a single image, painting every prefix worth painting. */
    async stream(url, generation) {
        const res = await fetch(url);
        if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
        const total = Number(res.headers.get('Content-Length')) || 0;
        const rate = Number(this.getAttribute('throttle')) || 0;
        let buf = new Uint8Array(total || 65536), have = 0, pending = false, lastPaint = 0;
        const reader = res.body.getReader();
        const t0 = performance.now();
        const repaint = () => {
            pending = false;
            if (generation !== this.generation) return;
            const r = decode(buf.subarray(0, have));
            if (r) this.paint(r.rgb, r.header.width, r.header.height);
            lastPaint = performance.now();
        };
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            if (generation !== this.generation) { reader.cancel(); return have; }
            // Feed large chunks through in slices when throttled, so the
            // progressive render is visible on a fast connection too.
            const step = rate ? Math.max(512, Math.floor(rate / 20)) : value.length;
            for (let off = 0; off < value.length; off += step) {
                const part = value.subarray(off, Math.min(value.length, off + step));
                if (have + part.length > buf.length) {
                    const grown = new Uint8Array(Math.max(buf.length * 2, have + part.length));
                    grown.set(buf.subarray(0, have));
                    buf = grown;
                }
                buf.set(part, have);
                have += part.length;
                if (rate) {
                    const due = t0 + (have / rate) * 1000;
                    const wait = due - performance.now();
                    if (wait > 0) await new Promise(r => setTimeout(r, wait));
                    repaint();
                } else if (!pending && performance.now() - lastPaint > 50) {
                    pending = true;
                    requestAnimationFrame(repaint);
                }
            }
        }
        repaint();
        return have;
    }
}

if (!customElements.get('nvdr-img')) customElements.define('nvdr-img', NvdrImg);
