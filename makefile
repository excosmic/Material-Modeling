FXN = cos,sin,sinc
TXT = filename.txt
FMT = jpeg,eps,pdf

.PHONY: plot write read

plot:
	python trigonometry.py --function=$(FXN) --print=$(FMT)
write:
	python trigonometry.py --function=$(FXN) --write=$(TXT)
read:
	python trigonometry.py --function=$(FXN) --read_from_file=$(TXT)
