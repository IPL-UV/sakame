# Reviewer Comments

## Editors Response

PONE-D-18-06167
A Note on Derivatives in Kernel Methods
PLOS ONE

Dear Mr Johnson,

Thank you for submitting your manuscript to PLOS ONE. After careful consideration, we have decided that your manuscript does not meet some of our criteria for publication and must therefore be rejected. In particular, we encourage the authors to clarify the major contribution of the paper and also perform experiments on real datasets to improve the technical standard of the manuscript.

I am sorry that we cannot be more positive on this occasion, but hope that you appreciate the reasons for this decision.

Yours sincerely,

Ana Roque
Academic Editor
PLOS ONE

---
## Reviewers' comments

Reviewer's Responses to Questions

Comments to the Author

1. Is the manuscript technically sound, and do the data support the conclusions?

The manuscript must describe a technically sound piece of scientific research with data that supports the conclusions. Experiments must have been conducted rigorously, with appropriate controls, replication, and sample sizes. The conclusions must be drawn appropriately based on the data presented. 

Reviewer #2: Partly
 

2. Has the statistical analysis been performed appropriately and rigorously? 

Reviewer #2: N/A
 

3. Have the authors made all data underlying the findings in their manuscript fully available?

The PLOS Data policy requires authors to make all data underlying the findings described in their manuscript fully available without restriction, with rare exception (please refer to the Data Availability Statement in the manuscript PDF file). The data should be provided as part of the manuscript or its supporting information, or deposited to a public repository. For example, in addition to summary statistics, the data points behind means, medians and variance measures should be available. If there are restrictions on publicly sharing data—e.g. participant privacy or use of data from a third party—those must be specified.

Reviewer #2: Yes
 

4. Is the manuscript presented in an intelligible fashion and written in standard English?

PLOS ONE does not copyedit accepted manuscripts, so the language in submitted articles must be clear, correct, and unambiguous. Any typographical or grammatical errors should be corrected at revision, so please note any specific errors here.

Reviewer #2: Yes
 

5. Review Comments to the Author

Please use the space provided to explain your answers to the questions above. You may also include additional comments for the author, including concerns about dual publication, research ethics, or publication ethics. (Please upload your review as an attachment if it exceeds 20,000 characters)

Reviewer #2: The paper analyses the derivatives of the models learnt by several kernel methods.
The topic of the paper is interesting, however it is not clear to me what is the contribution of this paper.
I believe the computation of the derivatives of different models and kernel functions is not new.
One of the motivations of this work, as stated in the Abstract and in the Introduction, is to give insights about the learned functions.
However, I think the paper fails in providing such insights.
Beginning of page 3: authors state that the derivatives can be related to the margin. The closest assessment of this sentence is in Section 3.3 and the corresponding Figure 2. The figure is a little unclear (authors don't clarify what is the y axis in the plots, I this it should be the norm of the learnt model W). In fact, the norm of W is related to the margin, but the paper doesn't clarify this.
I appreciate experiments on toy data for gaining insights on the theory. However, the aim of the proposed analysis is to gain insights on the learnt models, from real data, where such insights are actually needed.
For instance, the plots in Fig. 2 can be computed on real datasets, and the generalization performance of the learnt models varying the regularization can be assessed.
Section 6.4 looks also interesting, but it is very short and not clear enough.

I think the kind of analysis proposed by the paper is interesting, however there are some major shortcomings:
-The paper contribution is not clear
-Missing experiments on real-world datasets, that should assess if the proposed analysis may be applicable and useful to real data.

Typos:
-eq. 16, I think you meant g(f(x_*))
 

6. If you would like your identity to be revealed to the authors, please include your name here (optional).

Your name and review will not be published with the manuscript. 

Reviewer #2: (No Response)



[NOTE: If reviewer comments were submitted as an attachment file, they will be attached to this email and accessible via the submission site. Please log into your account, locate the manuscript record, and check for the action link "View Attachments". If this link does not appear, there are no attachment files to be viewed.]

- - - - -
For journal use only: PONEDEC3


---
## My Summary

In this section, I break down what the reviewer said and try to give my intial thoughts about what they said versus what we actually want to do.

* Contributions for the Paper are not clear
  > S/he claims that although the paper is interesting, s/he isn't clear what the contributions are from this paper. From my perspective, I really just want to link together the concepts of derivatives of kernel methods. So nothing super fancy but a small note (essentially an extended/glorified literature review with some personal ideas). However, I do think that the literative review lacked a coherent story of events. Instead it sounds a bit more like we just wanted to cover our asses with what we're talking about and make sure we cited everyone; which promotes a lost motivation.

* "Failed to given insights of the Learned Functions"
  > So the reviewer is stressing upon the idea that we did not insights. So maybe we weren't clear on what insights we would give. I don't want to stress on the data. I want to stress on the concepts. That basically the derivative has been studied in many applications but mainly from the point of view of their respective field (e.g. GP - Predictive Mean, SVM - Sensitivity, e.t.c.). But we want to take it a step further and promote the thinking of derivatives of any kernel function which could potentially mean something interesting to another kernel method.

* "derivatives can be related to the margin"
  > The reviewer points out that we don't verify the statement that the $||\omega||$ is related to the margin. We only state it. I guess this is a good point about why state a fact without giving any insight as to why or how. But this could also be attributed to people trying to talk about everything without explaining anything.

* "Y-Axis of the plot (section 3.3, figure 2.0)"
  > We didn't state what the y-axis is.

* "aim of the proposed analysis is to gain insights on the learnt models, from real data"
  > So...I have to read what we originally proposed because I don't remember us saying anything about the insights being on the data. If anything we just want to consolidate, clarify and motivate. Not actually do a full paper on data; that would be way too much.
    
  **TODO**: check where this statement came from and where in the abstract do we suggest this.
* Experiment for Figure 2 - " computed on real datasets, and the generalization performance of the learnt models varying the regularization can be assessed"
  > We could possibly do a full example to justify the regularization. But I think that is out of the scope of the paper. Again, to re-iterate, I don't want to do full-fledged examples because data is difficult to work with and would be an entire other paper in itself.
* "Section 6.4 is interesting, but it is very short and not clear enough."
  > He is referring to the section on Unfolding and Independization. This is indeed very short... To be honest, it really feels like an after-thought and I don't know how to rectify this. Actually, the whole HSIC section feels like an after-thought aside from the formulation. 
