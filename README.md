# Self Supervised Learning for Next-K-word-prediction-App

In the demo the predictions are't that accurate/coherant. This is because I used an MLP model which is a very basic neural network. This might not be able to learn long term dependencies between the words and hence the model dosen't understand the structure of the text very well unlike models like LSTMs, RNNs, Transformers or any other attention based models.

Also MLPs dont encode position inherently unlike RNNs and LSTMs. This might be another reason why the order of the words might appear weird. (The sentence might make sense if we reorder the words but otherwise seems random).

While positional encodings can be addeed to MLPs in various ways easily, for this demonstration I haven't used them to keep things simple, as I wanted to demonstrate how self-supervised learning can be used for next word 
prediction.


Self Supervised Learning: The idea is use only the "X" data(no ground truth labels). For example, in linear regression we map some x linearly to predict some y_pred and this should be close to the ground truth value y. But in the case of self-supervised learning we dont have such ground truth values. We use some part of the data to predict some other part of the data.

In this case of next k word prediction:
Data was created as:

Input: the quick brown fox -> Target: jumps
Input: quick brown fox jumps -> Target: over
Input: brown fox jumps over -> Target: the
Input: fox jumps over the -> Target: lazy
Input: jumps over the lazy -> Target: dog
Input: over the lazy dog -> Target: and
Input: the lazy dog and -> Target: runs
Input: lazy dog and runs -> Target: away
Input: dog and runs away -> Target: quickly

here a seq length of 3 is considered.

https://github.com/user-attachments/assets/478da950-898f-4518-8a55-572af800d6e5
