Final validation results (mean iou) after 15 epochs:

- linear:           0.42288503569130864
- convolutional:    0.4819981720540315
- transformer:      0.6201591706504092
- transformer-sh-qk:0.5969766314660988

1.3.1)
A transformer with shared queries and key performs slightly less worse than a normal transformer.
However it also has much less parameters, so this decrease in performance is justified.
Also it still performs much better than a linear or convolutional head.

Visually they don't look much different. When the normal transformer tends to color too much the Shared-Queries model tends to color not enough.

If a smaller model is desired the shared queries-keys seem like a valid option.

1.4)
Linear layer performs the worst, even worse than a convolutional layer when they have a similar amount of parameters.
This could be because of the fitting choice of a convolutional layer for an image. It gives the mdoel a slight advantage.

The transformer models perform much better but also have allot more parameters.
They should be chosen if the outcome is important and there are many resources to spend.

Overall they all scored to my surprise quite well which is due to the large prefitted model that they extend.

By analyzing the plots, it can be shown, that the transformers learn less smoothly compared to the linear layer.