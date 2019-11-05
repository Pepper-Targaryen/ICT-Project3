# Review: Web Service Injection Attack Detection

## Author: Victor Clincy and Hossain Shahriar 

## Institution: Kennesaw State University

## Date: ICITST-2017

## Reviewer: Yixiao FEI



​    To protect deployed web services against injection attacks, it is important to have defense techniques. Intrusion Detection Systems (IDS) are popular defense techniques to mitigate network layer attacks. This paper proposes an IDS for mitigating injection attacks on web services. We apply Genetic Algorithm (GA) as part of new attack signature generation for web services.

​    This paper has defined a chromosome representation for SOAP (Simple Object Access Protocol) log. Then the genetic algorithm is applied to generate the data set of attack logs by using two fitness functions (Hamming Distance and Levenshtein Distance). The initial evaluation shows, that HD performs better than LD.  Further, higher selection and mutation rates for crossing over chromosomes, the better the signatures are.

​    This paper develops a signature-based IDS for web services relying on GA to generate new attack signatures from a set of initial attack signatures. The paper didn't precise how to predict a new log as a normal access or an attack after generating the attack dataset.
