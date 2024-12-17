#!/bin/bash
# this trains the three individual networks with a range of fractions of the existing data provided to them
date

for i in {1..20}
do
	echo $i
	python ML_test_normal.py $i &> normaldetail$i.log &
done

wait
for i in {21..40}
do
	echo $i
	python ML_test_normal.py $i &> normaldetail$i.log &
done

wait
for i in {41..60}
do
	echo $i
	python ML_test_normal.py $i &> normaldetail$i.log &
done

wait
for i in {61..80}
do
	echo $i
	python ML_test_normal.py $i &> normaldetail$i.log &
done
wait
for i in {81..100}
do
	echo $i
	python ML_test_normal.py $i &> normaldetail$i.log &
done

wait

date

for i in {1..20}
do
	echo $i
	python ML_test_l.py $i &> ldetail$i.log &
done

wait
for i in {21..40}
do
	echo $i
	python ML_test_l.py $i &> ldetail$i.log &
done

wait
for i in {41..60}
do
	echo $i
	python ML_test_l.py $i &> ldetail$i.log &
done

wait
for i in {61..80}
do
	echo $i
	python ML_test_l.py $i &> ldetail$i.log &
done
wait
for i in {81..100}
do
	echo $i
	python ML_test_l.py $i &> ldetail$i.log &
done

wait

date

for i in {1..20}
do
	echo $i
	python ML_test_vl.py $i &> vldetail$i.log &
done

wait
for i in {21..40}
do
	echo $i
	python ML_test_vl.py $i &> vldetail$i.log &
done

wait
for i in {41..60}
do
	echo $i
	python ML_test_vl.py $i &> vldetail$i.log &
done

wait
for i in {61..80}
do
	echo $i
	python ML_test_vl.py $i &> vldetail$i.log &
done
wait
for i in {81..100}
do
	echo $i
	python ML_test_vl.py $i &> vldetail$i.log &
done

wait

date
