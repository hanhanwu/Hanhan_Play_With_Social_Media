
This project is to explore the existence of Underrated Answers in StackOverflow and to find whether it is possible to detect them automatically

<b>Underrated Answer</b> - The best solution at the time when I collect the data, but it has less votes than the top voted answer.

*******************************************************************************

OVERALL APPROACH

1. Data Collection
* Extratced 100,000+ StackOverflow (SO) urls from StackExahnge API
* Automatically found 2000+ SO posts with 3+ answers, and generated 2000+ JSON file to record Question, Answer, Comments data for each post
* Manually label the urls with "Y" or "N" to indicate whether there is an underrated answer. If it's "Y", reocrd the first 5 words of the Underrated Answer

2. Feature Generation
