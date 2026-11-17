# fluid-sim

SPH fluid simulation in WebGPU: a stream of 65,536 particles, computed entirely
on the GPU, as a single HTML file with no dependencies and no build step.

**Live:** https://nkuendig.github.io/fluid-sim/

Created with [Claude Fable 5.2](https://www.anthropic.com/claude/fable).

## Files

| File | Purpose |
|---|---|
| `index.html` | Landing page |
| `fluid.html` | The simulation in 3D: WGSL shaders, rendering, controls |
| `cpu_ref.mjs` | The same physics on the CPU, used to check the GPU |
| `fluid_2d.html`, `cpu_ref_2d.mjs` | The earlier 2D version |
| `fluid_before.html` | 2D version before the neighbour search was rewritten, for comparison |

## Running locally

`fluid.html` runs by double-click. Verify mode and the benchmark need a local
server because they load a module:

```bash
python3 -m http.server 8123
```

Then open `http://localhost:8123/`.

| Address | Purpose |
|---|---|
| `fluid.html?warm=12` | The stream is already full on the first frame |
| `fluid.html?points` | Starts in the particle view |
| `fluid.html?bench` | Measures milliseconds per physics step |
| `fluid.html?verify` | Compares GPU and CPU reference step by step |
| `node cpu_ref.mjs` | Fills the stream on the CPU and prints key figures |

## Browsers

WebGPU: Chrome or Edge 144+, Firefox 141+, Safari 26+.

## License

MIT, see [LICENSE](LICENSE).
