# MaskAI Jetson Setup
![Overall Schematic Jetson](/Jetson/Overall_Schematic_Jetson.png)

# Step 1 (PC/Jetson): Create an Azure ResourceGroup and Container Registry
1. Make sure you have [installed Azure CLI](https://docs.microsoft.com/zh-tw/cli/azure/install-azure-cli-windows?view=azure-cli-latest) on Windows
2. This [link](https://docs.microsoft.com/zh-tw/azure/container-registry/container-registry-get-started-azure-cli) outlines how to create an **Azure ResourceGroup** and **Azure Container Registry (ACR)** using CLI (you are fine once you can log into your ACR, you do not have to do everything in the tutorial)
3. Once **ACR** is created, go to your **ACR ==> Access Keys (tab)** on [Azure Portal](https://portal.azure.com), copy the **{Login Server}**, **{Username}** and **{Password}**; you will need these later
4. Also, take note of your **{resource group name}** for the ResourceGroup you created earlier

# Step 2 (Jetson): Create docker container image
1. Flash your Nvidia Jetson device with [JetPack 4.2.2](https://developer.nvidia.com/jetpack-422-archive) (follow the steps in this link, **will require an Ubuntu Host Computer**)
2. Make sure your Jetson Device is connected to the internet (cable preferably, WiFi not advised), and on the upper right corner of the Ubuntu Desktop, set powermode to **0: MAXN**
3. Open up ubuntu Terminal on our Jetson Device
4. Edit docker daemon.json:
```
sudo nano /etc/docker/daemon.json
```
5. add **"default-runtime": "nvidia"** so the whole thing looks like:
```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
6. Restart docker
```
sudo systemctl restart docker
```
7. Copy repository's entire "AI_AICare" folder into **/home/{$user}/**
8. Execute the following commands and input your **{Username}** and **{Password}** when prompted
```
cd /home/{$user}/AI_AICare
sudo docker login {Login Server}
sudo docker build -t {Login Server}/{image_name_of_your_choice} .
sudo docker push {Login Server}/{image_name_of_your_choice}
```

# Step 3: (PC/Jetson): Create an Azure IoTHub, IoTEdge Device and IoTEdge Device Module
1. Create an [Azure IoTHub](https://docs.microsoft.com/en-us/azure/iot-hub/iot-hub-create-using-cli), and remember your **{hub_name}**
2. Create an IoTEdge Device under the IoTHub using **Azure CLI** with this command: 
```
az iot hub device-identity create --hub-name {hub_name} --device-id myEdgeDevice --edge-enabled
```
3. Retrieve the **{CONNECTION_STRING}** for your EdgeDevice (this is very important) with this command: 
```
az iot hub device-identity show-connection-string --device-id myEdgeDevice --hub-name {hub_name}
```
4. Next, sign in to your [Azure Portal](https://portal.azure.com) and navigate to your IoTHub
5. On the left pane, select "IoT Edge" from the menu
6. Click on your Device ID (i.e. myEdgeDevice)
7. On the upper bar, select "Set Modules"
8. In the "Container Registry Settings" section of the page, provide your **{Login Server}** as the ADDRESS, **{Username}** as NAME and USER NAME, and **{Password}** as PASSWORD
![Container Registry Credentials](/icons/ContainerRegistryCredentials.png)
9.  In the IoT Edge Modules section of the page, select Add
10. Select "IoT Edge Module" from the drop-down menu, and provide your own module name, and in the "Image URI" section ENTER **{Login Server}/{image_name_of_your_choice}** (note that this is the container image that you "docker pushed" earlier)
![Setup Module](/icons/SetupModule.png)
11. Once that's done, select the "Container Create Options" tab and copy & paste these:
```json 
{
    "HostConfig": {
        "PortBindings": {
            "8001/tcp": [
                {
                    "HostPort": "8001"
                }
            ]
        }
    }
}
```
![ContainerCreateOptions](/icons/ContainerCreateOptions.png)

12.  Finally, press the "Update" button, then "Review + create" button and then press "Create" after; the Edge module should be up and running

# Step 4: (Jetson): Install Azure IoTEdge Runtime
1. Back on Jetson Device, open up ubuntu Terminal (make sure you have "curl" installed)
```
sudo apt-get update
sudo apt-get install curl
```
2. Execute the following to install iotedge runtime, moby-engine and moby-cli
```
curl https://packages.microsoft.com/config/ubuntu/18.04/multiarch/prod.list > ./microsoft-prod.list
sudo cp ./microsoft-prod.list /etc/apt/sources.list.d/
curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo cp ./microsoft.gpg /etc/apt/trusted.gpg.d/
sudo mv /var/lib/dpkg/info /var/lib/dpkg/info.bak
sudo mkdir /var/lib/dpkg/info
sudo apt update
sudo apt install -f moby-engine
sudo apt install -f moby-cli
sudo mv /var/lib/dpkg/info/* /var/lib/dpkg/info.bak
ls -a /var/lib/dpkg/info
sudo rm -rf /var/lib/dpkg/info
sudo mv /var/lib/dpkg/info.bak /var/lib/dpkg/info
sudo apt-get update
sudo apt-get install iotedge
sudo nano /etc/iotedge/config.yaml
```
3. search for and enter your previously retrieved **{CONNECTION_STRING}** into the quotations of this line: **device_connection_string: "*ADD DEVICE CONNECTION STRING HERE*"**
4. press ctrl+x, shift+y, ENTER
5. Now, restart iotedge and reboot jetson
```
sudo systemctl restart iotedge
sudo reboot
```
6. After Jetson Device has rebooted, wait for around 15min or so for iotedge, along with the modules running your container image to startup. You can check if everything is up and running by typing 
```
sudo iotedge list
```
7. Once everything is up and running (you should see 3 modules running, namely **edgeAgent**, **edgeHub** and **{your module name}**), go to the upper right corner of ubuntu desktop and set ipv4 address of your Jetson Device to "Manual" with IP as: **"192.168.99.95"**, and netmask as: **"24" or "255.255.255.0"**
8. reboot again and now the Face Mask Detection AI Server on your Jetson Device will run automatically
```
sudo reboot
```


