MIT License

Copyright (c) [2025] [Lum3on + ComfyUI_AudioTools Contributors]

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

--

Third-Party Software and Model Disclaimer

This project, ComfyUI Audio Toolkit (AudioTools), is licensed under the MIT License, as detailed in the LICENSE file.

However, this toolkit relies on several third-party libraries and pre-trained models which are distributed under their own, separate licenses. The MIT License for this project does not apply to these dependencies, and it is your responsibility as a user to understand and comply with their respective licensing terms.
Key Dependencies and Their Licenses

Below is a summary of the licenses for the primary dependencies used in this project. This list is for informational purposes and is not exhaustive. You should always verify the license of each component you use.

    Demucs (demucs):
        Code: The Demucs library source code is licensed under the MIT License.
        Pre-trained Models: The pre-trained models provided by the Demucs project (e.g., htdemucs, htdemucs_ft) are released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. This means they are not licensed for commercial use. Please review the terms of this license carefully if you intend to use the AI-based nodes (Stem Separator, Speech Denoise) for anything other than personal, non-commercial projects.

    Whisper (openai-whisper):
        The code and the models are licensed under the MIT License.

    PyTorch (torch, torchaudio):
        Licensed under a BSD-style license. It is generally permissive for commercial use.

    pyttsx3:
        Licensed under the BSD License.

    Librosa:
        Licensed under the ISC License, which is functionally equivalent to the MIT License.

    pyloudnorm:
        Licensed under the MIT License.

Your Responsibility
By using this ComfyUI custom node pack, you acknowledge that you are responsible for adhering to the licenses of all underlying software and models. The author of the ComfyUI Audio Toolkit is not liable for any licensing violations related to your use of these third-party dependencies.