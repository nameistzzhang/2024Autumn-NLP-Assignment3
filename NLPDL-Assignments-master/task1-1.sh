# for each mode run 10 loops
# python task1-1.py --mode="naive"
# python task1-1.py --mode="cache"
# python task1-1.py --mode="quantization"
# python task1-1.py --mode="qandc"

for i in {1..10}
do
    python task1-1.py --mode="naive"
done

for i in {1..10}
do
    python task1-1.py --mode="cache"
done

for i in {1..10}
do
    python task1-1.py --mode="quantization"
done

# for i in {1..10}
# do
#     python task1-1.py --mode="qandc"
# done

