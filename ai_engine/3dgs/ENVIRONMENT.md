# BrainDance 3DGS Engine - 统一环境与依赖文档

## 1. 系统与硬件基线

| 项目 | 实测值 |
|---|---|
| OS | `Ubuntu 22.04.5 LTS (Jammy)` |
| Kernel | `6.8.0-101-generic` |
| NVIDIA Driver | `570.211.01` |
| CUDA Driver API | `12.8` (`nvidia-smi`) |
| CUDA Toolkit | `12.8` (`nvcc 12.8.93`) |
| GPU | `NVIDIA L20 x2` (46GB each) |
| Conda 环境名 | `Braindance` |
| Conda 前缀目录 | `/home/jiangbeihu/miniconda3/envs/Braindance` |
| Python 实际路径 | `/usr/bin/python` |
| Python 版本 | `3.10.12` |
| pip 实际路径 | `/home/jiangbeihu/.local/bin/pip` |
| site-packages | `/home/jiangbeihu/.local/lib/python3.10/site-packages` |
| PyTorch | `2.10.0+cu128` |

## 2. 工具依赖（系统级，非 pip）

| 工具 | 路径 | 版本/状态 | GPU 状态 | 是否按规则视为“需单独编译” |
|---|---|---|---|---|
| COLMAP | `/usr/local/bin/colmap` | `3.14.0.dev0 (Commit 01ca9ec6)` | `with CUDA` | 是 |
| GLOMAP | `/usr/local/bin/glomap` | 可用 | `compiled with CUDA` | 是 |
| FFmpeg | `/usr/bin/ffmpeg` | `4.4.2-0ubuntu0.22.04.1` | `cuda + nvenc` 可用 | 按项目规则：GPU 版视为是 |
| Nerfstudio CLI | `/home/jiangbeihu/.local/bin/ns-train` | 来自 `nerfstudio==1.1.5` | 依赖 PyTorch CUDA | 否（Python 包） |

说明：你要求“需要 GPU 加速的工具视为需要编译”，本文按该规则标注。

## 3. Python 依赖结论

- `conda env export -n Braindance` 基本为空壳，实际依赖由 `pip` 层维护。
- 全量 Python 包：`582` 个（见第 7 节完整列表）。

## 4. 需要单独编译/源码安装的 Python 依赖

### 4.1 Editable 本地源码安装

- `-e git+https://github.com/tianxingleo/BrainDance.git@5191f585fca8bba0a14bee583ad207df314ab340#egg=sam3d_objects&subdirectory=ai_engine/3dgs/src/libs/sam-3d-objects`
- `-e git+https://github.com/tianxingleo/BrainDance.git@5191f585fca8bba0a14bee583ad207df314ab340#egg=sharp&subdirectory=ai_engine/3dgs/src/libs/ml-sharp`

### 4.2 Git 源码安装

- `clip @ git+https://github.com/ultralytics/CLIP.git@88ade288431a46233f1556d1e141901b3ef0a36b`
- `moge @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`
- `utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900`

### 4.3 二进制扩展依赖（wheel 不匹配时需要本地编译）

- `cumm-cu121==0.7.11`
- `faiss-cpu==1.13.2`
- `gsplat==1.4.0`
- `manifold3d==3.4.0`
- `nerfacc==0.5.2`
- `numpy==1.26.4`
- `nvidia-cublas-cu12==12.8.4.1`
- `nvidia-cuda-cupti-cu12==12.8.90`
- `nvidia-cuda-nvcc-cu12==12.1.105`
- `nvidia-cuda-nvrtc-cu12==12.8.93`
- `nvidia-cuda-runtime-cu12==12.8.90`
- `nvidia-cudnn-cu12==9.10.2.21`
- `nvidia-cufft-cu12==11.3.3.83`
- `nvidia-cufile-cu12==1.13.1.3`
- `nvidia-curand-cu12==10.3.9.90`
- `nvidia-cusolver-cu12==11.7.3.90`
- `nvidia-cusparse-cu12==12.5.8.93`
- `nvidia-cusparselt-cu12==0.7.1`
- `nvidia-nccl-cu12==2.27.5`
- `nvidia-nvjitlink-cu12==12.8.93`
- `nvidia-nvshmem-cu12==3.4.5`
- `nvidia-nvtx-cu12==12.8.90`
- `open3d==0.18.0`
- `opencv-python==4.9.0.80`
- `opencv-python-headless==4.10.0.84`
- `pandas==2.3.3`
- `point-cloud-utils==0.29.5`
- `pyarrow==23.0.1`
- `pycocotools==2.0.7`
- `pycolmap==3.13.0`
- `pymeshlab==2025.7.post1`
- `pynvml==13.0.1`
- `rawpy==0.26.1`
- `scipy==1.15.3`
- `spconv-cu121==2.3.8`
- `torch==2.10.0`
- `torchaudio==2.10.0`
- `torchvision==0.25.0`
- `triton==3.6.0`
- `vtk==9.6.0`
- `xformers==0.0.35`

## 5. 不需要本地编译的依赖

- 除第 4 节列出的依赖外，其余依赖通常为纯 Python 或通用 wheel，可直接安装。

## 6. nerfstudio patch

- patch 文件：`patches/0002_nerfstudio_eval_utils_weights_only.patch`
- 目的：将 `torch.load(load_path, map_location="cpu")` 改为 `torch.load(..., weights_only=False)`。

应用命令：

```bash
conda activate Braindance
SITE_PACKAGES=$(python -c "import site; print(site.getusersitepackages())")
cd "$SITE_PACKAGES"
patch -p0 < /path/to/BrainDance/patches/0002_nerfstudio_eval_utils_weights_only.patch
```

验证命令：

```bash
python -c "import importlib.util, pathlib; p=pathlib.Path(importlib.util.find_spec('nerfstudio.utils.eval_utils').origin); print('weights_only=False' in p.read_text())"
```

## 7. 全量依赖列表（全部包，不拆分）

```text
absl-py==2.4.0
accelerate==1.12.0
addict==2.4.0
aiofiles==24.1.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
alabaster==0.7.12
annotated-doc==0.0.4
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.12.1
appdirs==1.4.4
apturl==0.5.2
argcomplete==3.6.3
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
astor==0.8.1
asttokens==3.0.1
async-lru==2.2.0
async-timeout==4.0.3
attrs==23.2.0
audioread==3.1.0
auto_gptq==0.7.1
autoflake==2.3.1
av==12.0.0
azure-core==1.38.2
azure-identity==1.25.2
azure-storage-blob==12.28.0
azure-storage-file-datalake==12.23.0
babel==2.18.0
bcrypt==3.2.0
beautifulsoup4==4.10.0
beniget==0.4.1
bidict==0.23.1
bitsandbytes==0.43.0
black==24.3.0
bleach==6.3.0
blinker==1.9.0
boto3==1.42.59
botocore==1.42.59
braceexpand==0.1.7
Brlapi==0.8.3
brotli==1.2.0
cachetools==6.2.6
calmsize==0.1.3
ccimport==0.4.4
certifi==2020.6.20
cffi==2.0.0
cfgv==3.5.0
cftime==1.5.2
chardet==4.0.0
charset-normalizer==3.4.4
circuitbreaker==2.1.3
click==8.3.1
clip @ git+https://github.com/ultralytics/CLIP.git@88ade288431a46233f1556d1e141901b3ef0a36b
cloudpickle==3.1.2
colorama==0.4.6
colorcet==2.0.2
coloredlogs==15.0.1
colorlog==6.10.1
comet_ml==3.56.0
comm==0.2.3
command-not-found==0.3
commonmark==0.9.1
conda-pack==0.7.1
ConfigArgParse==1.7.1
configobj==5.0.9
contourpy==1.3.2
cramjam==2.11.0
crcmod==1.7
cryptography==46.0.5
cuda-bindings==12.9.4
cuda-pathfinder==1.4.0
cuda-python==12.1.0
cumm-cu121==0.7.11
cupshelpers==1.0
cycler==0.11.0
cyclopts==4.6.0
Cython==3.2.4
dash==4.0.0
dashscope==1.25.13
dataclasses==0.6
datasets==4.6.1
dbus-python==1.2.18
debugpy==1.8.20
decorator==4.4.2
decord==0.6.0
defer==1.0.6
defusedxml==0.7.1
deprecation==2.1.0
descartes==1.1.0
dill==0.4.0
distlib==0.4.0
distro==1.7.0
distro-info==1.1+ubuntu0.2
dnspython==2.8.0
docker==7.1.0
docopt==0.6.2
docstring_parser==0.17.0
docutils==0.17.1
dulwich==0.25.2
duplicity==0.8.21
e3nn==0.6.0
easydict==1.13
einops==0.8.2
einops-exts==0.0.4
embreex==2.17.7.post7
everett==3.1.0
evo==1.34.3
exceptiongroup==1.2.0
executing==2.2.1
faiss-cpu==1.13.2
fastapi==0.135.1
fastavro==1.9.4
fasteners==0.19
fastjsonschema==2.21.2
fenics-basix==0.10.0.post0
fenics-dolfinx==0.10.0.post2
fenics-ffcx==0.10.1
fenics-ufl==2025.2.0.post0
ffmpy==1.0.0
filelock==3.25.0
fire==0.7.1
flake8==7.0.0
Flask==3.0.3
fonttools==4.29.1
fpsample==1.0.0
fqdn==1.5.1
freetype-py==2.5.1
frozenlist==1.8.0
fs==2.4.12
fsspec==2025.12.0
ftfy==6.2.0
future==0.18.2
fvcore==0.1.5.post20221221
gast==0.5.2
gdown==5.2.0
gekko==1.3.2
gitdb==4.0.12
GitPython==3.1.46
glcontext==3.0.0
google-api-core==2.30.0
google-auth==2.48.0
google-cloud-core==2.5.0
google-cloud-storage==2.10.0
google-crc32c==1.8.0
google-pasta==0.2.0
google-resumable-media==2.8.0
googleapis-common-protos==1.72.0
gradio==6.8.0
gradio_client==2.2.0
groovy==0.1.2
grpcio==1.78.0
gsplat==1.4.0
gyp==0.1
h11==0.16.0
h2==4.3.0
h5py==3.12.1
h5py.-debian-h5py-serial==3.6.0
hdfs==2.7.3
hf-xet==1.3.2
hpack==4.1.0
html5lib==1.1
httpcore==1.0.9
httplib2==0.22.0
httpx==0.28.1
huggingface_hub==0.36.2
humanfriendly==10.0
hydra-core==1.3.2
hydra-submitit-launcher==1.2.0
hyperframe==6.1.0
identify==2.6.17
idna==3.3
igraph==0.11.8
ImageIO==2.37.2
imageio-ffmpeg==0.6.0
imagesize==1.3.0
imath==0.0.2
importlib-metadata==4.6.4
iniconfig==2.3.0
iopath==0.1.10
ipykernel==7.2.0
ipython==8.38.0
ipywidgets==8.1.8
isodate==0.7.2
isoduration==20.11.0
itsdangerous==2.2.0
jaxtyping==0.3.7
jedi==0.19.2
jeepney==0.7.1
Jinja2==3.1.6
jiter==0.13.0
jmespath==1.1.0
joblib==1.5.3
json5==0.13.0
jsonlines==4.0.0
jsonpickle==3.0.4
jsonpointer==2.4
jsonschema==4.22.0
jsonschema-specifications==2025.9.1
jupyter==1.1.1
jupyter-console==6.6.3
jupyter-events==0.12.0
jupyter-lsp==2.3.0
jupyter_client==8.8.0
jupyter_core==5.9.1
jupyter_server==2.17.0
jupyter_server_terminals==0.5.4
jupyterlab==4.5.5
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
jupyterlab_widgets==3.0.16
keyring==23.5.0
kiwisolver==1.3.2
language-selector==0.1
lark==1.3.1
launchpadlib==1.10.16
lazr.restfulclient==0.14.4
lazr.uri==1.0.6
lazy_loader==0.4
libcst==1.8.6
librosa==0.10.1
lightning==2.3.3
lightning-utilities==0.15.3
llvmlite==0.46.0
lockfile==0.12.2
loguru==0.7.2
louis==3.20.0
lxml==4.8.0
lz4==3.1.3+dfsg
macaroonbakery==1.3.1
Mako==1.1.3
manifold3d==3.4.0
mapbox_earcut==2.0.0
Markdown==3.10.2
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.8
matplotlib-inline==0.2.1
mccabe==0.7.0
mdurl==0.1.2
mediapy==1.2.6
meshio==5.3.0
mistune==3.2.0
mmh3==5.2.0
mock==4.0.3
moderngl==5.12.0
moge @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b
monotonic==1.6
more-itertools==8.10.0
moreorless==0.5.0
mosaicml-streaming==0.7.5
moviepy==2.2.1
mpi4py==3.1.3
mpmath==1.3.0
msal==1.35.0
msal-extensions==1.3.1
msgpack==1.1.2
msgpack-numpy==0.4.8
msgspec==0.20.0
multidict==6.7.1
multiprocess==0.70.18
mypy_extensions==1.1.0
nanobind==2.9.2
narwhals==2.17.0
natsort==8.4.0
nbclient==0.10.4
nbconvert==7.17.0
nbformat==5.10.4
nerfacc==0.5.2
nerfstudio==1.1.5
nest-asyncio==1.6.0
netCDF4==1.5.8
netifaces==0.11.0
networkx==3.4.2
ninja==1.13.0
nodeenv==1.10.0
notebook==7.5.4
notebook_shim==0.2.4
numba==0.64.0
numexpr==2.14.1
numpy==1.26.4
nuscenes-devkit==1.2.0
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvcc-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-ml-py==13.590.48
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
nvidia-pyindex==1.0.9
oauthlib==3.2.0
objsize==0.7.0
oci==2.168.0
olefile==0.46
omegaconf==2.3.0
open3d==0.18.0
openai==2.24.0
opencv-python==4.9.0.80
opencv-python-headless==4.10.0.84
OpenEXR==3.3.3
opt-einsum-fx==0.1.4
opt_einsum==3.4.0
optimum==1.18.1
optree==0.14.1
orjson==3.10.0
overrides==7.7.0
packaging==26.0
Panda3D==1.10.16
panda3d-gltf==1.2.1
panda3d-simplepbr==0.13.1
pandas==2.3.3
pandocfilters==1.5.1
param===None
parameterized==0.9.0
paramiko==3.5.1
parso==0.8.6
pathos==0.3.4
pathspec==1.0.4
pccm==0.4.16
pdoc3==0.10.0
peft==0.10.0
petsc4py==3.15.1
pexpect==4.8.0
pi_heif==1.3.0
pillow==11.3.0
pillow_heif==1.3.0
pip-system-certs==4.0
platformdirs==4.9.2
plotly==6.6.0
pluggy==1.6.0
ply==3.11
plyfile==1.1.3
point-cloud-utils==0.29.5
polars==1.38.1
polars-runtime-32==1.38.1
polyscope==2.3.0
pooch==1.9.0
portalocker==3.2.0
postgrest==2.28.0
pox==0.3.7
ppft==1.7.8
pre_commit==4.5.1
proglog==0.1.12
prometheus_client==0.24.1
prompt_toolkit==3.0.52
propcache==0.4.1
proto-plus==1.27.1
protobuf==3.20.3
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
pusimp==0.1.0
pyarrow==23.0.1
pyasn1==0.6.2
pyasn1_modules==0.4.2
pybind11==3.0.2
pycairo==1.20.1
pycocotools==2.0.7
pycodestyle==2.11.1
pycollada==0.9.3
pycolmap==3.13.0
pycparser==2.21
pyct==0.4.7a3
pycups==2.0.1
pydantic==2.12.5
pydantic_core==2.41.5
pydot==1.4.2
pydub==0.25.1
pyflakes==3.2.0
pyglet==2.1.13
Pygments==2.19.2
PyGObject==3.42.1
pyiceberg==0.11.1
PyJWT==2.11.0
pyliblzfse==0.4.1
pymacaroons==0.13.0
pymeshfix==0.17.0
pymeshlab==2025.7.post1
pymongo==4.6.3
PyNaCl==1.5.0
pyngrok==7.5.0
pynvml==13.0.1
PyOpenGL==3.1.0
pyOpenSSL==25.3.0
pyparsing==3.3.2
pypose==0.7.5
pyquaternion==0.9.9
pyrender==0.1.45
pyRFC3339==1.1
pyroaring==1.0.3
PySocks==1.7.1
pytest==8.1.1
python-apt==2.4.0+ubuntu4.1
python-box==6.1.0
python-dateutil==2.9.0.post0
python-debian==0.1.43+ubuntu1.1
python-discovery==1.1.0
python-dotenv==1.2.2
python-engineio==4.13.1
python-json-logger==4.0.0
python-multipart==0.0.22
python-pycg==0.9.2
python-snappy==0.7.3
python-socketio==5.16.1
pythran==0.10.0
pytorch-lightning==2.6.1
pytorch-msssim==1.0.0
pytz==2022.1
pyvista==0.47.1
pyxdg==0.27
PyYAML==6.0.3
pyzmq==27.1.0
randomname==0.2.1
rawpy==0.26.1
realtime==2.28.0
referencing==0.37.0
regex==2026.2.28
reportlab==3.6.8
requests==2.32.5
requests-toolbelt==1.0.0
retrying==1.4.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rich==14.3.3
rich-rst==1.3.2
roma==1.5.1
roman==3.3
rootutils==1.0.7
rosbags==0.11.0
rouge==1.0.1
rpds-py==0.30.0
rsa==4.9.1
Rtree==1.3.0
ruamel.yaml==0.19.1
s3transfer==0.16.0
safehttpx==0.1.7
safetensors==0.7.0
sagemaker==2.242.0
sagemaker-core==1.0.77
-e git+https://github.com/tianxingleo/BrainDance.git@5191f585fca8bba0a14bee583ad207df314ab340#egg=sam3d_objects&subdirectory=ai_engine/3dgs/src/libs/sam-3d-objects
schema==0.7.8
scikit-image==0.23.1
scikit-learn==1.7.2
scikit_build_core==0.12.1
scipy==1.15.3
scooby==0.11.0
screen-resolution-extra==0.0.0
screeninfo==0.8.1
seaborn==0.13.2
SecretStorage==3.3.1
semantic-version==2.10.0
Send2Trash==2.1.0
sentence-transformers==2.6.1
sentencepiece==0.2.1
sentry-sdk==2.54.0
setproctitle==1.3.7
shapely==2.0.7
-e git+https://github.com/tianxingleo/BrainDance.git@5191f585fca8bba0a14bee583ad207df314ab340#egg=sharp&subdirectory=ai_engine/3dgs/src/libs/ml-sharp
shellingham==1.5.4
simple-websocket==1.1.0
simplejson==3.19.2
six==1.16.0
slepc4py==3.15.1
smdebug-rulesconfig==1.0.1
smmap==5.0.2
smplx==0.1.28
sniffio==1.3.1
snowballstemmer==2.2.0
socksio==1.0.0
soundfile==0.13.1
soupsieve==2.3.1
soxr==1.0.0
spconv-cu121==2.3.8
Sphinx==4.3.2
splines==0.3.0
ssh-import-id==5.11
stack-data==0.6.3
starlette==0.52.1
stdlibs==2026.2.26
storage3==2.28.0
StrEnum==0.4.15
strictyaml==1.7.3
submitit==1.5.4
supabase==2.28.0
supabase-auth==2.28.0
supabase-functions==2.28.0
svg.path==7.0
sympy==1.14.0
systemd-python==234
tabulate==0.9.0
tblib==3.2.2
tenacity==9.1.4
tensorboard==2.16.2
tensorboard-data-server==0.7.2
tensorly==0.9.0
termcolor==3.3.0
terminado==0.18.1
texttable==1.7.0
threadpoolctl==3.6.0
tifffile==2025.5.10
timm==0.6.7
tinycss2==1.4.0
tokenizers==0.15.2
toml==0.10.2
tomli==2.0.1
tomlkit==0.13.3
torch==2.10.0
torch_fidelity==0.4.0
torchaudio==2.10.0
torchmetrics==1.8.2
torchvision==0.25.0
tornado==6.5.4
tqdm==4.67.3
trailrunner==1.4.0
traitlets==5.14.3
transformers==4.39.3
trimesh==4.11.2
triton==3.6.0
typeguard==4.5.1
typer==0.24.1
typing-inspection==0.4.2
typing_extensions==4.15.0
tyro==1.0.8
tzdata==2025.3
ubuntu-drivers-common==0.0.0
ubuntu-pro-client==8001
ufoLib2==0.13.1
ufw==0.36.1
ultralytics==8.4.19
ultralytics-thop==2.0.18
unattended-upgrades==0.1
unicodedata2==14.0.0
UNKNOWN==0.0.0
uri-template==1.3.0
urllib3==2.6.3
usb-creator==0.3.7
usd-core==26.3
usort==1.0.8.post1
utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900
uvicorn==0.41.0
vhacdx==0.0.10
virtualenv==21.1.0
viser==0.2.7
vtk==9.6.0
wadler_lindig==0.1.7
wadllib==1.3.6
wandb==0.20.0
wcwidth==0.2.14
webcolors==1.13
webdataset==0.2.86
webencodings==0.5.1
websocket-client==1.9.0
websockets==15.0.1
Werkzeug==3.0.6
widgetsnbextension==4.0.15
wrapt==2.1.1
wsproto==1.3.2
wurlitzer==3.1.1
xatlas==0.0.9
xdg==5
xformers==0.0.35
xkit==0.0.0
xxhash==3.6.0
yacs==0.1.8
yarl==1.23.0
yourdfpy==0.0.60
zipp==1.0.0
zstandard==0.25.0
zstd==1.5.7.3
```

## 8. 一次性核验命令

```bash
uname -a
cat /etc/os-release
nvidia-smi | head -n 15
nvcc --version | head -n 4
which colmap && colmap -h | head -n 3
which glomap && glomap --help | head -n 4
which ffmpeg && ffmpeg -version | head -n 3
ffmpeg -hwaccels | head -n 20
ffmpeg -encoders | rg -i 'nvenc|cuvid|cuda'
which ns-train && ns-train -h | head -n 2
python -m pip freeze | wc -l
```
