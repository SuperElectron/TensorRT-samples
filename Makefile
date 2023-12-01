CONTAINER=trt
ARCH_BIN='6.1;7.5'

ifeq ($(shell uname -m),x86_64)
 	ARCH=amd
 else
 	ARCH=aarch64
endif

clean_git:
	@echo "Cleaning up merged branches"
	git branch -D `git branch --merged | grep -v \* | xargs`
	@echo "Match local (laptop) to remote (github)"
	git remote prune origin

build:
	docker buildx build -t $(CONTAINER) --build-arg ARCH=$(ARCH) .
start:
	docker run -it --rm --name $(CONTAINER) \
		--privileged \
		--net=host \
		--user=0 \
		--runtime nvidia \
		--gpus all \
		-e DISPLAY=$(DISPLAY) \
		-v /tmp/.X11-unix/:/tmp/.X11-unix \
		-v ~/.Xauthority:/root/.Xauthority \
		-v "/dev:/dev" \
		-v `pwd`/src:/src \
		-w /src \
		$(CONTAINER) bash
