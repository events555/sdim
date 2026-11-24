# shepherd
sdim's large-scale study tool: qudit error-correction experiments built, sampled with `sdim.dem`,
decoded in parallel, and turned into threshold plots, with resumable results files.

shepherd is documented in the [shepherd section of sdim's README](https://github.com/events555/sdim#shepherd-large-scale-studies).
Quick start, from the root of the sdim repository:

```bash
pip install -e .                    # sdim
pip install -e "./shepherd[plot]"   # shepherd
shepherd list                       # built-in studies
```
