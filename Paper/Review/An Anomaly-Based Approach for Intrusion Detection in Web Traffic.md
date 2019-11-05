# An Anomaly-Based Approach for Intrusion Detection in Web Traffic

## Authors

CARMEN TORRANO-GIMENEZ, ALEJANDRO PEREZ-VILLEGAS, GONZALO ALVAREZ

## Conference

Journal of Information Assurance and Security, vol. 5, no. 4, pp. 446–454, 2010

## Reviewer

Lei WANG

## Resume

> The system relies on a XML file to classify the incoming requests as normal or anomalous. When the XML file has enough information, it can characterize the normal behavior of the target web application.

The XML file contains HTTP verbs, HTTP headers, accessed resources, arguments, and value for the arguments.

**Detection process flow**

* Verb Check
* Headers Check
* Resource check
* All arguments allowed for the resource check
* All mandatory arguments present
* Argument values check

By analyzing the XML file, as long as the XML file correctly defines, a normal request are obtained.
