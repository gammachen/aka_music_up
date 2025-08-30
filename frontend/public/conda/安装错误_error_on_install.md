```shell
for install https://github.com/TachibanaYoshino/AnimeGANv3?tab=readme-ov-file

python3.8 3.9 3.11均不好使

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for numpy
Failed to build numpy
ERROR: Failed to build installable wheels for some pyproject.toml based projects (numpy)


(animeganv3_with_py311) shhaofu@shhaofudeMacBook-Pro AnimeGANv3 % pip install numpy==1.21.5
ERROR: Could not find a version that satisfies the requirement numpy==1.21.5 (from versions: 1.3.0, 1.4.1, 1.5.0, 1.5.1, 1.6.0, 1.6.1, 1.6.2, 1.7.0, 1.7.1, 1.7.2, 1.8.0, 1.8.1, 1.8.2, 1.9.0, 1.9.1, 1.9.2, 1.9.3, 1.10.0.post2, 1.10.1, 1.10.2, 1.10.4, 1.11.0, 1.11.1, 1.11.2, 1.11.3, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 1.13.3, 1.14.0, 1.14.1, 1.14.2, 1.14.3, 1.14.4, 1.14.5, 1.14.6, 1.15.0, 1.15.1, 1.15.2, 1.15.3, 1.15.4, 1.16.0, 1.16.1, 1.16.2, 1.16.3, 1.16.4, 1.16.5, 1.16.6, 1.17.0, 1.17.1, 1.17.2, 1.17.3, 1.17.4, 1.17.5, 1.18.0, 1.18.1, 1.18.2, 1.18.3, 1.18.4, 1.18.5, 1.19.0, 1.19.1, 1.19.2, 1.19.3, 1.19.4, 1.19.5, 1.20.0, 1.20.1, 1.20.2, 1.20.3, 1.21.0, 1.21.1, 1.22.0, 1.22.1, 1.22.2, 1.22.3, 1.22.4, 1.23.0, 1.23.1, 1.23.2, 1.23.3, 1.23.4, 1.23.5, 1.24.0, 1.24.1, 1.24.2, 1.24.3, 1.24.4, 1.25.0, 1.25.1, 1.25.2, 1.26.0, 1.26.1, 1.26.2, 1.26.3, 1.26.4, 2.0.0, 2.0.1, 2.0.2, 2.1.0rc1, 2.1.0, 2.1.1, 2.1.2, 2.1.3, 2.2.0rc1, 2.2.0, 2.2.1, 2.2.2, 2.2.3, 2.2.4)
ERROR: No matching distribution found for numpy==1.21.5

pip install numpy==1.19.5

      creating build/temp.macosx-11.0-arm64-3.11/private/var/folders/9l/kbk_mdlj0x5bcvscm_41pmbm0000gn/T/pip-install-ibegrld5/numpy_0558c3682ca14352a6bc4bfac3514fc7/numpy/_build_utils/src
      compile options: '-DNPY_INTERNAL_BUILD=1 -DHAVE_NPY_CONFIG_H=1 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE=1 -D_LARGEFILE64_SOURCE=1 -DNO_ATLAS_INFO=3 -DHAVE_CBLAS -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/umath -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/npymath -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/common -Inumpy/core/include -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/include/numpy -Inumpy/core/src/common -Inumpy/core/src -Inumpy/core -Inumpy/core/src/npymath -Inumpy/core/src/multiarray -Inumpy/core/src/umath -Inumpy/core/src/npysort -I/opt/anaconda3/envs/animeganv3_with_py311/include/python3.11 -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/common -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/npymath -c'
      extra options: '-faltivec -I/System/Library/Frameworks/vecLib.framework/Headers'
      clang: numpy/core/src/multiarray/alloc.c
      clang: numpy/core/src/multiarray/common.c
      clang: numpy/core/src/multiarray/buffer.c
      clang: numpy/core/src/multiarray/conversion_utils.c
      clang: numpy/core/src/multiarray/array_assign_scalar.c
      clang: numpy/core/src/multiarray/descriptor.c
      clang: numpy/core/src/multiarray/datetime_strings.c
      clang: build/src.macosx-11.0-arm64-3.11/numpy/core/src/multiarray/einsum.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: numpy/core/src/multiarray/hashdescr.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: numpy/core/src/multiarray/nditer_constr.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: numpy/core/src/multiarray/refcount.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: numpy/core/src/multiarray/multiarraymodule.c
      clang: numpy/core/src/multiarray/scalarapi.c
      clang: build/src.macosx-11.0-arm64-3.11/numpy/core/src/multiarray/lowlevel_strided_loops.c
      clang: numpy/core/src/multiarray/temp_elide.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: numpy/core/src/multiarray/vdot.c
      clang: build/src.macosx-11.0-arm64-3.11/numpy/core/src/umath/loops.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: numpy/core/src/umath/ufunc_object.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: numpy/core/src/umath/ufunc_type_resolution.c
      clang: build/src.macosx-11.0-arm64-3.11/numpy/core/src/npymath/ieee754.c
      clang: numpy/core/src/common/ucsnarrow.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: numpy/core/src/common/array_assign.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: build/src.macosx-11.0-arm64-3.11/numpy/core/src/common/npy_cpu_features.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: /private/var/folders/9l/kbk_mdlj0x5bcvscm_41pmbm0000gn/T/pip-install-ibegrld5/numpy_0558c3682ca14352a6bc4bfac3514fc7/numpy/_build_utils/src/apple_sgemv_fix.c
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      clang: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
      error: Command "clang -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/anaconda3/envs/animeganv3_with_py311/include -arch arm64 -fPIC -O2 -isystem /opt/anaconda3/envs/animeganv3_with_py311/include -arch arm64 -DNPY_INTERNAL_BUILD=1 -DHAVE_NPY_CONFIG_H=1 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE=1 -D_LARGEFILE64_SOURCE=1 -DNO_ATLAS_INFO=3 -DHAVE_CBLAS -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/umath -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/npymath -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/common -Inumpy/core/include -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/include/numpy -Inumpy/core/src/common -Inumpy/core/src -Inumpy/core -Inumpy/core/src/npymath -Inumpy/core/src/multiarray -Inumpy/core/src/umath -Inumpy/core/src/npysort -I/opt/anaconda3/envs/animeganv3_with_py311/include/python3.11 -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/common -Ibuild/src.macosx-11.0-arm64-3.11/numpy/core/src/npymath -c numpy/core/src/multiarray/alloc.c -o build/temp.macosx-11.0-arm64-3.11/numpy/core/src/multiarray/alloc.o -MMD -MF build/temp.macosx-11.0-arm64-3.11/numpy/core/src/multiarray/alloc.o.d -faltivec -I/System/Library/Frameworks/vecLib.framework/Headers" failed with exit status 1
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for numpy
Failed to build numpy
ERROR: Failed to build installable wheels for some pyproject.toml based projects (numpy)


(animeganv3_with_py311) shhaofu@shhaofudeMacBook-Pro AnimeGANv3 % conda install -c conda-forge numpy=1.21.6 scikit-image=0.19.0 opencv=4.6.0

Channels:
 - conda-forge
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: failed

LibMambaUnsatisfiableError: Encountered problems while solving:
  - package scikit-image-0.19.0-py310hdead3df_0 requires python >=3.10,<3.11.0a0 *_cpython, but none of the providers can be installed

Could not solve for environment specs
The following packages are incompatible
├─ pin on python 3.11.* =* * is installable and it requires
│  └─ python =3.11 *, which can be installed;
└─ scikit-image =0.19.0 * is not installable because there are no viable options
   ├─ scikit-image 0.19.0 would require
   │  └─ python >=3.10,<3.11.0a0 *_cpython, which conflicts with any installable versions previously reported;
   ├─ scikit-image 0.19.0 would require
   │  └─ python >=3.8,<3.9.0a0 *_cpython, which conflicts with any installable versions previously reported;
   └─ scikit-image 0.19.0 would require
      └─ python >=3.9,<3.10.0a0 *_cpython, which conflicts with any installable versions previously reported.

Pins seem to be involved in the conflict. Currently pinned specs:
 - python=3.11

Collecting onnxruntime==1.10.0 (from -r requirements.txt (line 7))
  Downloading http://mirrors.cloud.aliyuncs.com/pypi/packages/8b/74/5b30ee067fba2228d0b6bbf5c8915cadc5ce98b4c97beba978b03128d344/onnxruntime-1.10.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 20.5 MB/s eta 0:00:00
ERROR: Ignored the following versions that require a different python version: 0.22.0 Requires-Python >=3.9; 0.22.0rc1 Requires-Python >=3.9; 0.23.0 Requires-Python >=3.10; 0.23.0rc0 Requires-Python >=3.10; 0.23.0rc2 Requires-Python >=3.10; 0.23.1 Requires-Python >=3.10; 0.23.2 Requires-Python >=3.10; 0.23.2rc1 Requires-Python >=3.10; 0.24.0 Requires-Python >=3.9; 0.24.0rc1 Requires-Python >=3.9; 0.25.0 Requires-Python >=3.10; 0.25.0rc0 Requires-Python >=3.10; 0.25.0rc1 Requires-Python >=3.10; 0.25.0rc2 Requires-Python >=3.10; 0.25.1 Requires-Python >=3.10; 0.25.2 Requires-Python >=3.10; 0.25.2rc0 Requires-Python >=3.10
ERROR: Could not find a version that satisfies the requirement onnxruntime-gpu==1.1.0 (from versions: 1.11.0, 1.11.1, 1.12.0, 1.12.1, 1.13.1, 1.14.0, 1.14.1, 1.15.0, 1.15.1, 1.16.0, 1.16.1, 1.16.2, 1.16.3, 1.17.0, 1.17.1, 1.18.0, 1.18.1, 1.19.0, 1.19.2)
ERROR: No matching distribution found for onnxruntime-gpu==1.1.0


        dist.fetch_build_eggs(dist.setup_requires)
      WARNING: The repository located at mirrors.cloud.aliyuncs.com is not a trusted or secure host and is being ignored. If this repository is available via HTTPS we recommend you use HTTPS instead, otherwise you may silence this warning and allow it anyway with '--trusted-host mirrors.cloud.aliyuncs.com'.
      ERROR: Could not find a version that satisfies the requirement pytest-runner (from versions: none)
      ERROR: No matching distribution found for pytest-runner
      Traceback (most recent call last):
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/setuptools/installer.py", line 107, in _fetch_build_egg_no_warn
          subprocess.check_call(cmd)
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/subprocess.py", line 369, in check_call
          raise CalledProcessError(retcode, cmd)
      subprocess.CalledProcessError: Command '['/root/miniconda3/envs/animeganv3_with_py310/bin/python', '-m', 'pip', '--disable-pip-version-check', 'wheel', '--no-deps', '-w', '/tmp/tmpnllspje3', '--quiet', '--index-url', 'http://mirrors.cloud.aliyuncs.com/pypi/simple/', 'pytest-runner']' returned non-zero exit status 1.

      The above exception was the direct cause of the following exception:

      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-c18d8nee/onnx_4e125be867804f17a895451a897c8eaf/setup.py", line 339, in <module>
          setuptools.setup(
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/setuptools/__init__.py", line 116, in setup
          _install_setup_requires(attrs)
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/setuptools/__init__.py", line 89, in _install_setup_requires
          _fetch_build_eggs(dist)
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/setuptools/__init__.py", line 94, in _fetch_build_eggs
          dist.fetch_build_eggs(dist.setup_requires)
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/setuptools/dist.py", line 663, in fetch_build_eggs
          return _fetch_build_eggs(self, requires)
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/setuptools/installer.py", line 44, in _fetch_build_eggs
          resolved_dists = pkg_resources.working_set.resolve(
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/pkg_resources/__init__.py", line 888, in resolve
          dist = self._resolve_dist(
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/pkg_resources/__init__.py", line 924, in _resolve_dist
          dist = best[req.key] = env.best_match(
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/pkg_resources/__init__.py", line 1262, in best_match
          return self.obtain(req, installer)
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/pkg_resources/__init__.py", line 1298, in obtain
          return installer(requirement) if installer else None
        File "/root/miniconda3/envs/animeganv3_with_py310/lib/python3.10/site-packages/setuptools/installer.py", line 109, in _fetch_build_egg_no_warn
          raise DistutilsError(str(e)) from e
      distutils.errors.DistutilsError: Command '['/root/miniconda3/envs/animeganv3_with_py310/bin/python', '-m', 'pip', '--disable-pip-version-check', 'wheel', '--no-deps', '-w', '/tmp/tmpnllspje3', '--quiet', '--index-url', 'http://mirrors.cloud.aliyuncs.com/pypi/simple/', 'pytest-runner']' returned non-zero exit status 1.
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.

ERROR: Could not find a version that satisfies the requirement onnxruntime-gpu==1.1.0 (from versions: 1.11.0, 1.11.1, 1.12.0, 1.12.1, 1.13.1, 1.14.0, 1.14.1, 1.15.0, 1.15.1, 1.16.0, 1.16.1, 1.16.2, 1.16.3, 1.17.0, 1.17.1, 1.18.0, 1.18.1, 1.19.0, 1.19.2)
ERROR: No matching distribution found for onnxruntime-gpu==1.1.0



```


