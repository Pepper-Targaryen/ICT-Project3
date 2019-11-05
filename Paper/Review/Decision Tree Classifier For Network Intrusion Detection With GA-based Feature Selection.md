# Decision Tree Classifier For Network Intrusion Detection With GA-based Feature Selection

## Authors

GARY STEIN, BING CHEN, ANNIE S.WU, KIEN A.HUA

## Conference

 [ACM Southeast Regional Conference (2) 2005](https://dblp.uni-trier.de/db/conf/ACMse/ACMse2005-2.html#SteinCWH05): 136-141 

## Reviewer

**Lei WANG**

## Resume

> This paper uses a genetic algorithm to select a subset of input features for decision tree classifiers, with a goal of increasing the detection rate and decreasing the false alarm rate in network intrusion detection.

GA-based Feature Selection algorithm is based on the wrapper model. Related work: [[ Automatic Parameter Selection by Minimizing Estimated Error ](https://pdf.sciencedirectassets.com/308842/3-s2.0-C20090277051/3-s2.0-B9781558603776500451/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEAaCXVzLWVhc3QtMSJGMEQCIFWDH5knmNKytmjXpcZHDZPKGYus1dZbAD8WyXXTfevTAiAywomok2nS9YCs46XUz49C7PdWWi%2F0mXlrZZWk%2Fbe3tirPAghYEAIaDDA1OTAwMzU0Njg2NSIM7iSuhtrWTmt5i1vOKqwCUiFERt3TDdgs60q2YRQSwRvbePVFslVoik%2BxbT7cTK76ARRf%2FwraprH4N3AZ7LDqHjHq2kgLV%2FiNJn0uZ2vTEFfPZUsl81xNzSzUfTut%2FLBMtWFMMbX%2BQ1gk9lk4otJVlj9%2BY32GY8Z7hWrLDaB%2BlT22letPsDP6LKSQtIEYp6DqVurcy%2B0kI5PK6viZ%2F5csqA4xzx%2BkBXZ8ATb1XJb4XftAU3%2BgcoWWNWQTjxSrO1T%2B%2BRyYM8pQyuxLU0UhUKCXXHVixwrftdvEnNgi5D5yES9kcR6BjLRA4ip%2FH0lIEiflSaoEWCgIl6dr2YBQgROOgIcm4cRXAAWgXRCOLq4P7nJ34lDzc71RrKSxmO1lDY%2BSFQRasX4NDyWwB2Lw6k%2BHTjK2A0bqCr94tj2MMMm%2BhO4FOtECTJdbpY88nfDcuVAWHz%2BOsrEpU%2FY9IEJzN1ooL6D07PuEVoNmFURpCHpTi0BBTgwT%2BZQqKPParav4aRQV50jfSM0x1hwGt82G1oBFy%2BdBWuvWJ0XwNjlavSxdMso3CMi4IdzO7VlxvrPtFqvDeEw1XanbOXIAUFxgy%2FC6RVbo%2Frm959JGBh20z%2FU%2BKvoAGbff1l%2B9gfJOJcGf2epg%2BVG7Bzzvj%2B9e%2Fua%2BWxfbYoooo0M8TH7TRkk7nOO8snYDhF%2BB35FaQw88s8mDuVSP8P5iO6PykVeIuBoAd187ALY5ZVmZZqHCvtagF80cToDPaH1IWKMZog6%2B29l0eZt1%2FnTAwiwFHaW3Mhub9BhhiWr0a65QG8%2FL5KVFPMEjwpxRHzy1KakqgwNIAOYoysd8cp9bpAwr0lXE42hBSxJQNf%2FuOvjkeRlSQIONIdkR03DZn04WtA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191105T074155Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQZOBXOM5%2F20191105%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=4f2c5332e9a4fd3289ff6c86ff42ce2138553e32b72e9de05553b5fee85bd3ff&hash=cbebacab981bc13703cf7df5c2bd809a112fe45b8996dce3209ed03fe6d93c5c&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=B9781558603776500451&tid=spdf-b3328d92-607c-4b5e-9e7e-9e8ee0154224&sid=fb1164687e39394f7399942710484fcaaa7agxrqa&type=client)]

**GA/Decision Tree Hybrid**

* Generate the initial population (Each individual represent a choice of available features)
* C4.5 program (Decision Tree Constructor)
* Decision Tree Evaluator
* Fitness computation (Error rate)
* Generate next generation (100 generations)
* Build final decision tree classifier
