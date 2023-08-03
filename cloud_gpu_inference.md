::: {.cell .markdown}
# Using cloud servers for GPU-based inference

:::


::: {.cell .markdown}

Machine learning models are most often trained in the "cloud", on powerful centralized servers with specialized resources (like GPU acceleration) for training machine learning models.  These servers are also well-resources for inference, i.e. making predictions on new data.

In this experiment, we will use a cloud server equipped with GPU acceleration for fast inference in an image classification context.

:::

::: {.cell .markdown}

This notebook assumes you already have a "lease" available for an RTX6000 GPU server on the CHI@UC testbed. Then, it will show you how to:

* launch a server using that lease
* attach an IP address to the server, so that you can access it over SSH
* install some fundamental machine learning libraries on the server
* use a pre-trained image classification model to do inference on the server
* delete the server

:::

::: {.cell .markdown}
## Launch a cloud server

We will start by preparing our environment in this notebook, then launching a cloud server using our pre-existing lease.

:::

::: {.cell .markdown}

First, we load some required libraries:

:::

::: {.cell .code}
``` python
import chi
from chi import server
from chi import lease
import datetime
import os
```
:::

::: {.cell .markdown}

We indicate that we're going to use the CHI@UC site. We also need to specify the name of the Chameleon "project" that this experiment is part of. The project name will have the format "CHI-XXXXXX", where the last part is a 6-digit number, and you can find it on your [user dashboard](https://chameleoncloud.org/user/dashboard/).

In the cell below, replace the project ID with your *own* project ID, then run the cell.

:::

::: {.cell .code}
``` python
chi.use_site("CHI@UC")
chi.set("project_name", "CHI-XXXXXX")
```
:::

::: {.cell .markdown}

Next, we'll specify the lease ID. This notebook assumes you already have a "lease" for an RTX6000 GPU server on CHI@UC. To get the ID of this lease,

* Vist the CHI@UC ["reservations" page](chi.uc.chameleoncloud.org/project/leases/).
* Click on the lease name.
* On the following page, look for the value next to the word "Id" in the "Lease" section.

Fill in the lease ID inside the quotation marks in the following cell, then run the cell.


:::

::: {.cell .code}
``` python
lease_id ="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```
:::

::: {.cell .markdown}

Now, we are ready to launch a server! 

:::

::: {.cell .markdown}

First, we'll specify the name for our server - we'll include our username and the experiment name in the server name, so that it will be easy to identify our server in the CHI@UC web interface.

:::

::: {.cell .code}
``` python
username = os.environ.get("USER")
expname = "cloud-gpu"
# set a name for the server
server_name = f"{username}-{expname}".replace("_","-")
```
:::


::: {.cell .markdown}

We will specify the disk image that we want our server to use.

:::


::: {.cell .code}
```python
chi.set("image", "CC-Ubuntu20.04")
```
:::



::: {.cell .markdown}

Then, we can create the server! 

:::

::: {.cell .code}
``` python
server.create_server(
    server_name, 
    reservation_id=lease.get_node_reservation(lease_id),
    image_name=chi.get("image")
)
server_id = server.get_server_id(server_name)
```
:::

::: {.cell .markdown}

The next cell waits for the server to be active.

:::

::: {.cell .code}
```python
# wait until server is ready to use
server.wait_for_active(server_id)
```
:::



::: {.cell .markdown}

Once the server is created, you should be able to see it and monitor its status on the [CHI@UC web interface](https://chi.uc.chameleoncloud.org/project/instances/). (If there was any problem while starting the server, you can also delete the server instance from that interface, in order to be able to try again.)

:::

::: {.cell .markdown}
## Attach an address and access your server over SSH

:::

::: {.cell .markdown}

Next, we will attach an address to our server, then use SSH to access its terminal.

First, we'll attach an address:

:::

::: {.cell .code}
``` python
public_ip = server.associate_floating_ip(server_id)
server.wait_for_tcp(public_ip, port=22)
```
:::

::: {.cell .markdown}

Now we can open a terminal in the Jupyter interface to access the server over SSH, using the SSH command that is printed by the following cell:

:::

::: {.cell .code}
``` python
print("ssh cc@%s" % public_ip)
```
:::

::: {.cell .markdown}

## Install machine learning libraries on your server

:::


::: {.cell .code}
```python
from chi import ssh

node = ssh.Remote(public_ip)
```
:::


::: {.cell .code}
```python
node.run('sudo apt update')
node.run('sudo apt -y install python3-pip python3-dev')
node.run('sudo pip3 install --upgrade pip')
```
:::


::: {.cell .code}
```python
node.run('wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb')
node.run('sudo dpkg -i cuda-keyring_1.0-1_all.deb')
node.run('sudo apt update')
node.run('sudo apt -y install linux-headers-$(uname -r)')
node.run('sudo apt-mark hold cuda-toolkit-12-config-common nvidia-driver-535') # don't let it install this cuda
node.run('sudo apt -y install nvidia-driver-520') # this driver likes CUDA 11.8
```
:::


::: {.cell .code}
```python
try:
    node.run('sudo reboot') # reboot and wait for it to come up
except:
    pass
server.wait_for_tcp(public_ip, port=22)
node = ssh.Remote(public_ip) 
```
:::


::: {.cell .code}
```python
node.run('sudo apt -y install cuda-11-8 cuda-runtime-11-8 cuda-drivers=520.61.05-1')
node.run('sudo apt -y install nvidia-gds-11-8') # install instructions say to do this separately!
node.run('sudo apt -y install libcudnn8=8.9.3.28-1+cuda11.8 nvidia-cuda-toolkit') # make sure the get cuda-11-8 version
```
:::


::: {.cell .code}
```python
node.run("echo 'PATH=\"/usr/local/cuda-11.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin\"' | sudo tee /etc/environment")
```
:::


::: {.cell .code}
```python
try:
    node.run('sudo reboot')
except:
    pass
server.wait_for_tcp(public_ip, port=22)
node = ssh.Remote(public_ip) # note: need a new SSH session to get new PATH
node.run('nvidia-smi')
node.run('nvcc --version')
```
:::


::: {.cell .code}
```python
node.run('python3 -m pip install --user Cython==0.29.32')
node.run('wget https://raw.githubusercontent.com/teaching-on-testbeds/colab/main/requirements_chameleon_dl.txt -O requirements_chameleon_dl.txt')
node.run('python3 -m pip install --user -r requirements_chameleon_dl.txt --extra-index-url https://download.pytorch.org/whl/cu113 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html')
```
:::



::: {.cell .markdown}
Test your installation - make sure Tensorflow can see the GPU:
:::


::: {.cell .code}
```python
node.run('python3 -c \'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))\'')
# should say: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
```
:::


::: {.cell .markdown}
## Install additional libraries for fast inference on GPU

To squeeze all the inference speed we can out of our GPU, we'll also install TensorRT, a framework that optimizes inference speed on NVIDIA's GPUs.

:::

::: {.cell .code}
```python
# hold libcudnn8 so we won't try to upgrade it to an incompatible version
node.run('sudo apt-mark hold libcudnn8')
# specify python-cuda to match installed cuda version
node.run('python3 -m pip install --user cuda-python==11.8.0')
# specify versions for ALL the tensorrt components
node.run('sudo apt -y install tensorrt-dev=8.6.0.12-1+cuda11.8 libnvinfer-dev=8.6.0.12-1+cuda11.8 libnvinfer-lean-dev=8.6.0.12-1+cuda11.8 libnvinfer-headers-dev=8.6.0.12-1+cuda11.8 libnvinfer8=8.6.0.12-1+cuda11.8 libnvinfer-lean8=8.6.0.12-1+cuda11.8 libnvinfer-dispatch-dev=8.6.0.12-1+cuda11.8 libnvinfer-plugin-dev=8.6.0.12-1+cuda11.8 libnvinfer-vc-plugin-dev=8.6.0.12-1+cuda11.8 libnvparsers-dev=8.6.0.12-1+cuda11.8 libnvonnxparsers-dev=8.6.0.12-1+cuda11.8 libnvparsers8=8.6.0.12-1+cuda11.8 libnvinfer-vc-plugin8=8.6.0.12-1+cuda11.8 libnvinfer-dispatch8=8.6.0.12-1+cuda11.8 libnvinfer-headers-plugin-dev=8.6.0.12-1+cuda11.8 libnvinfer-plugin8=8.6.0.12-1+cuda11.8 libnvonnxparsers8=8.6.0.12-1+cuda11.8 libcudnn8-dev=8.9.3.28-1+cuda11.8')
node.run('sudo apt -y install python3-libnvinfer-dev=8.6.0.12-1+cuda11.8 python3-libnvinfer=8.6.0.12-1+cuda11.8 python3-libnvinfer-lean=8.6.0.12-1+cuda11.8 python3-libnvinfer-dispatch=8.6.0.12-1+cuda11.8')
# need to update tensorflow to one that is linked against tensorrt8
node.run('python3 -m pip install --user tensorflow==2.12.0')
```
:::


::: {.cell .markdown}
## Transfering files to the server

Later in this notebook, we'll run an image classification model - a model that accepts an image as input and "predicts" the name of the object in the image - on the server. To do this, we'll need to upload some files to the server:

* a sample image
* Python code to load a model and make a prediction on the image

These are all contained in the `image_model` directory. We can upload them to the server using `scp`, and specify the source directory (in the Jupyter environment) and destination directory (on the server).

(The Python code will directly download a pre-trained model and the associated list of labels when we run it.)


:::

::: {.cell .code}
``` python
!scp -r -i ~/work/.ssh/id_rsa -o StrictHostKeyChecking=no image_model cc@{public_ip}:~/
```
:::


::: {.cell .markdown}
## Use a pre-trained image classification model to do inference

Now, we can use the materials we uploaded to the server, and do inference - make a prediction - *on* the server. 


In this example, we will use a machine learning model that is specifically designed for fast inference using GPU acceleration. We will ask it to make a prediction for the following image:

:::


::: {.cell .code}
```python
from IPython.display import Image
Image('image_model/parrot.jpg') 
```
:::

::: {.cell .code}
```python
node.run('python /home/cc/image_model/model.py')
```
:::

::: {.cell .markdown}

Make a note of the time it took to generate the prediction - would this inference time be acceptable for all applications? Also make a note of the model's three best "guesses" regarding the label of the image - is the prediction accurate?

:::

::: {.cell .markdown}
## Use a pre-trained image classification model to do inference with optimizations

Now we willl repeat the image classification above, but with a version of the model that is compiled with TensorRT, for extra optimizations on NVIDIA GPUs.

:::


::: {.cell .markdown}
First, we'll convert the same model to a TensorRT equivalent - this will take a while.
:::


::: {.cell .code}
```python
node.run('python /home/cc/image_model/model-convert.py')
```
:::


::: {.cell .markdown}
Then, we can run the optimized version of the model - 
:::

::: {.cell .code}
```python
node.run('python /home/cc/image_model/model-opt.py')
```
:::



::: {.cell .markdown}

Make a note of the time it took to generate the prediction - how does this compare to the previous one?

:::


::: {.cell .markdown}
## Delete the server

Finally, we should stop and delete our server so that others can create new servers using the same lease. To delete our server, we can run the following cell:

:::

::: {.cell .code}
```python
chi.server.delete_server(server_id)
```
:::


::: {.cell .markdown}
Also free up the IP that you we attached to the server, now that it is no longer in use:
:::



::: {.cell .code}
```python
ip_details = chi.network.get_floating_ip(public_ip)
chi.neutron().delete_floatingip(ip_details["id"])
```
:::
