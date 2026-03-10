## Build Draco for 3DGS

```bash
git clone --recursive https://github.com/Rajrup/DracoGS.git DracoGS
cd DracoGS

conda activate mesongs
pip install git+https://github.com/fraunhoferhhi/PLAS.git
pip install torchpq cupy
pip install imageio
```