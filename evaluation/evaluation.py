from contextualized_topic_models.evaluation.measures import CoherenceNPMI,CrosslingualRetrieval,TopicDiversity
from contextualized_topic_models.models.multilingual_contrast import MultilingualContrastiveTM
topics=
model1=CoherenceNPMI(topics=topics,texts=texts)
model2=CrosslingualRetrieval(topics=topics)
model3=TopicDiversity(topics=topics)