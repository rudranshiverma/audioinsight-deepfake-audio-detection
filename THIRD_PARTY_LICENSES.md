# Third-party code

## ASVspoof 2019 t-DCF and EER code

`audio_utils.py` contains `obtain_asv_error_rates`, `compute_det_curve`, `compute_eer` and `compute_tDCF`, adapted from the official ASVspoof 2019 t-DCF Python package (`eval_metrics.py`, version 1.0). Numerical code is unchanged; printing was removed and `sys.exit` calls were replaced by exceptions.

These four functions are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses/by-nc-sa/4.0/), as is the original package. They may not be used commercially.

## RawBoost

`rawboost.py` is adapted from https://github.com/TakHemlata/RawBoost-antispoofing (`RawBoost.py`). The only change makes `randRange` return a scalar for NumPy 2 compatibility.

MIT License

Copyright (c) 2021 Hemlata

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
