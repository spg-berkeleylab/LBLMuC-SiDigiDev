# VSCode Configuration

## Getting Docker
Currently there are only instructions for UNIX systems and Windows through Windows Subsystem for Linux(WSL).

Linux/Mac:
https://docs.docker.com/get-docker/

Windows w/ WSL:
https://docs.docker.com/docker-for-windows/wsl/

## Helpful Extensions

The following extensions VSCode are useful for development in containers.

### (WSL only) Remote-WSL
Allows you to run `code` in your WSL environment in order to run VSCode with its full functionality.

VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl

### ROOT Viewer
Allows you to view ROOT files through VSCode, instead of initializing TBrowsers in ROOT.

More info: https://root.cern/blog/vscode-extension-announcement/

### Remote-Containers
Allows you to work in a container in VSCode. Terminals opened in this state will be inside the container.

More info: https://code.visualstudio.com/docs/remote/containers.

## Important VSCode Workspace Files
The most important files are a ``.vscode/c_cpp_properties.json`` and ``.devcontainer.json`` file.

### CPP Properties
Make sure to include libraries that you want to use in this file so VSCode IntelliSense functions properly. Most notably, ROOT libraries are typically stored in: `/usr/include/root/`

### Dev Container
This is the file that the Remote-Containers extension latches onto. This will allow the extension to run the Docker container as well as launch VSCode inside of it for development. Here is an example of how one may look.

```json
{
	"image": "infnpd/mucoll-ilc-framework:1.5-centos8",

	"containerEnv": { "DISPLAY":"${DISPLAY}",
					  "USER":"${localEnv:USER}",
					  "HOME":"/home/${localEnv:USER}",
					  "MUC_CONTAINER_VER":"1.5-centos8"					  
					},
	"mounts": [
		"source=/cvmfs,target=/cvmfs,type=bind,consistency=cached",
		"source=${localEnv:HOME}/.Xauthority,target=/home/${localEnv:USER}/.Xauthority,type=bind,consistency=cached",
	],
	"workspaceMount": "source=path/to/working/directory,target=/home/${localEnv:USER},type=bind,consistency=delegated",
	"workspaceFolder": "/home/${localEnv:USER}/LBLMuC-SiDigiDev",

	"runArgs": ["--net=host"]
}
```
where the source tag is replaced with your working directory.