build: 
	docker build -t demianight/algency_test .
run:
	docker run --name algency_test -p 8000:8000 demianight/algency_test
kill:
	docker rm -f algency_test
