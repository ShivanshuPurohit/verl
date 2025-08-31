phony: run1 run2 net

TAG?="verl-test2"
VERL_PATH?="${PWD}/../"

net:
	sudo docker network create test

run1:
	docker container rm -f c1
	docker run -it --name c1 --network test --gpus '"device=0,1,2,3"' --ipc host -v"$(VERL_PATH)":/verl_repo $(TAG) bash

run2:
	docker container rm -f c2
	docker run -it --name c2 --network test --gpus '"device=4,5,6,7"' --ipc host -v"$(VERL_PATH)":/verl_repo $(TAG) bash

