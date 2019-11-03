# A Cross Platform Intrusion Detection System using Inter Server Communication Technique
## Authors

Ms. R.Priyadarshini, Ms. Jagadiswaree.D, Ms. Fareedha.A, Mr. Janarthanan.M
## Conference
International Conference on Recent Trends in Information Technology 2011
## Resume

It uses a detection system PHPIDS for detecting cross site attacks. It mainly aims to 2
types of intrusions: SQLi (SQL injection) and XSS (Cross-site scripting).
SQLi exploits the dynamically generated SQL statements for client input and can extract
sensitive data from the database of compromised sites. XSS involves execution of script or
malicious code on the user’s web browser thus allowing access to any cookies, sessions etc.
It contains the following fi ve phases:
* Detection intrusion by PHPIDS APIs.
* Preventing the attacks using IDS.
* Logging the attacks into Database. It contains id, server, language, database, and label
of vulnerability. It can also include timestamps.
* Mining the web log using tree traversal.
* Generate report.

## Advantages:

This model can not only detect and prevent intrusions in PHP but also be used for other
languages such as .NET and JSP. At last the report is generated for futher analysis so that
the developers can have better understanding of intrusions and optimize it.

## Disadvantages:
The preventing system is ready-made and no innovation on it. The experiments shows how
many attacks have been detected and prevented. However, there is no baseline and we cannot
even know the performance of this model.
