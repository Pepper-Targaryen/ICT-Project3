# A Generalized Feature Extraction Scheme to Detect 0-Day Attacks via IDS Alerts

## Authors
Jungsuk SONG, Hiroki TAKAKURA, Yongjin KWON

## Conference
International Symposium on Applications and the Internet 2008

## Resume

In this article, it focuses on the 0-day attacks. 0-day attack means an unknown attack.
The ultimate challenge in intrusion detection ﬁeld is how we can exactly identify such an attack.
The author first extract 7 features from records and then use One-class SVM to do classification.
Firstly, for extracting features, we have several observations on 0-day attacks:

* It tends to attack against only a certain port.
* It takes long-period of times until 0-day attacks in the wild.
* Even if situation of network changes moment by moment, the behavior of 0-day attacks quite irregular with well-known attacks.
* 0-day attacks tend to be performed by only one group rather than many ones.
* 0-day attacks induce many different kinds of alerts.

Considering above, the author gives seven features for each record:

* NUM SAME SA DA DP: Among N alerts, the number of alerts whose destination address is the same to the current alert.
* RATE DIFF ALERT SAME SA DA DP: Rate of the number of alerts whose alert types are different from the current alert to n.
* TIME STDDEV SAME SA DA DP: Standard deviation of the time intervals between each instance of n alerts including the current alert.
* NUM SAME SA DP DIFF DA: Among N alerts, the number of alerts whose destination address is different from the current alert.
* RATE DIFF ALERT SAME SA DP DIFF DA: Rate of the number of alerts whose alert types are different from the current alert to (N − n). 
* TIME STDDEV SAME SA DP DIFF DA: Standard deviation of the time intervals between each 
instance of (N − n) alerts including the current alert.
* RATE REVERSE SP SAME SA DP: Rate of the number of the alerts whose source port is the same or larger than that of the current alert to N.

Then we can detect attacks using One-class SVM according to these 7 features.
