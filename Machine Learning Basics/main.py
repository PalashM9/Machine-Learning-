#!/usr/bin/env python3
import json

import numpy as np
import random
import pandas as pd
from tabulate import tabulate
import warnings
from collections import defaultdict
from typing import Callable, Optional, Tuple
from collections import Counter
#suppress warnings
warnings.filterwarnings('ignore')

from typing import Callable, Optional
########################################################################
#             Concept learning of web site functions.
########################################################################

# Symbol for contradiction
CONTRADICTION = "!"

def g_0(n: int) -> 'tuple[str, ...]':
    """Returns the maximally-general hypothesis for an n-dimensional feature
    space."""
    return ("?",) * n


def s_0(n: int) -> 'tuple[str, ...]':
    """Returns the maximally-specific hypothesis for an n-dimensional feature
    space."""
    return (CONTRADICTION,) * n


def more_general(h1: 'tuple[str, ...]', h2: 'tuple[str, ...]') -> bool:
    """Returns True iff. hypothesis h1 is more-general-than hypothesis h2.
    """
    assert len(h1) == len(h2), \
        f"can't compare {h1} and {h2} with different number of dimensions."

   #### Declaring empty array ####

    hypothesis_one = h1
    hypothesis_two = h2
    more_general = []
    is_more_general = False
    if_more_general = False

    #### Iterating over individual indexes of h1 & h2 ####
    for indexInH1,indexInH2 in ((hypothesis_one[indexOfHypothesis],hypothesis_two[indexOfHypothesis]) for indexOfHypothesis in range(min(len(h1), len(h2)))):
        """ if indexed element in h1 has '?' assiging value as true, if indexed element in h2 has '?' 
        assigning value as false, and if the indexed element in h2 has '0' assigning value as true"""

       # is_more_general = indexInH1 == "?" or (indexInH1 != "0" and (indexInH1 == indexInH2 or indexInH2 == "0" or indexInH2 == "!"))
        if indexInH1 == "?":
            is_more_general = True
        elif indexInH1 != "0" and (indexInH1 == indexInH2 or indexInH2 == "0" or indexInH2 == "!"):
            is_more_general = True
        else:
            is_more_general = False
        more_general.append(is_more_general)

    """Using 'all' function to find if a single false value is present and deducing that the h1 is less general than h2
    which fails to satisfy all the above condition"""
    if_more_general = all(more_general)

    return if_more_general

def fulfills(example, hypothesis):
    """Returns True iff. the example fulfills the hypothesis, i.e. hypothesis(example) = True"""
    ### Note: the implementation is the same as for the more_general relation.
    return more_general(hypothesis, example)


def min_generalizations(h: 'tuple[str, ...]', x: 'tuple[str, ...]') -> 'list[tuple[str, ...]]':
    """Returns all minimal generalizations of hypothesis h that are fulfilled by
    example x.
    """
    assert len(h) == len(x), \
        f"can't generalize {h} with respect to {x} with different number of dimensions."

    #cross-checking if 'x' is not more general than 'h'
    examples = x
    hypothesis = h
    is_general = fulfills(examples,hypothesis)
    # Taking the hypothesis as a new list for return
    new_generalized_h = list(hypothesis)
    for indexed_hyp in range(len(hypothesis)):
        if not is_general:
            if hypothesis[indexed_hyp] != '0' and hypothesis[indexed_hyp] != '!' and hypothesis[indexed_hyp] != examples[indexed_hyp]:
                new_generalized_h[indexed_hyp] = '?'
            else:
                new_generalized_h[indexed_hyp] = examples[indexed_hyp]

    new_generalized_h = [tuple(new_generalized_h)]
    return new_generalized_h

def maximum_generalized_hypothesis(n: int) -> 'tuple[str, ...]':
    return ("?",) * n

def maximum_specified_hypothesis(n: int) -> 'tuple[str, ...]':
    return (CONTRADICTION,) * n

def find_domains(examples: 'list[tuple[str, ...]]') -> 'list[list[str]]':
    """Returns the list of possible values for every attribute of the given examples.
    """
    
    #Empty array with all domains
    domain_array = []
    attribute_index_domains = []

    for indexed_example in range(len(examples[0])):
        for sub_indexed_example in range(len(examples)):
            val = examples[sub_indexed_example][indexed_example]
            # helping to add unique elements
            check_repetation = val not in attribute_index_domains
            if check_repetation:
               attribute_index_domains.append(val)

        domain_array.append(attribute_index_domains)
        attribute_index_domains = []

    return (list(domain_array))


def min_specializations(h: 'tuple[str, ...]', domains: 'list[list[str]]',
                        x: 'tuple[str, ...]') -> 'list[tuple[str, ...]]':
    """Returns the minimal specializations of h, using the given feature domains, that do not admit x.
    """
    assert len(h) == len(x), \
        f"can't specialize {h} with respect to {x} with different number of dimensions."
    assert len(h) == len(domains), \
        f"can't specialize {h} with respect to {domains} with different number of dimensions."
    
    #Taking empty array list
    min_specializations_list = []
    domain_attributes = domains
    general_hypothesis = list(h)

    for index_of_hypothesis in range(len(general_hypothesis)):
        for unique_domain in domain_attributes[index_of_hypothesis]:
            if x[index_of_hypothesis] != unique_domain:
               general_hypothesis[index_of_hypothesis-1] = '?'
               general_hypothesis[index_of_hypothesis] = unique_domain
               min_specializations_list.append(tuple(general_hypothesis))

    return list(tuple(min_specializations_list))

def candidate_elimination(examples: 'list[tuple[str, ...]]', classes: 'list[bool]',
                          domains: 'list[list[str]]') -> 'tuple[set[tuple[str, ...]], set[tuple[str, ...]]]':
    """Runs the candidate elimination algorithm on the given training examples
    and true class values, using the given attribute domains when specializing
    hypotheses.
    Returns the sets S and G.
    """
    assert len(examples) == len(classes), \
        "Need exactly as many true class values as examples!"

    #Taking dynamic length for different domains over feature attributes
    length_of_domains = len(domains)

    ##########      Step 1 -> G = {g0} & S = {s0}      ##############
    generalization_set = set([maximum_generalized_hypothesis(length_of_domains)])
    specialization_set = set([maximum_specified_hypothesis(length_of_domains)])
    example_set = examples

    #converting tuple to list for indexing purpose
    classes_list = list(classes)


    for feature_attribute in example_set:
        #Acquiring index of the YES/NO or Target Concept - here we have classes as TRUE OR FALSE
        index_of_feature = example_set.index(feature_attribute)

        # Applying slide-pseudo analysis of keeping generalized values as per the fulfills function
        # when the indexed_feature is a positive example.
        if classes_list[index_of_feature]:
            specialization_list = list(specialization_set)
            generalization_set = {generalized_feature for generalized_feature in generalization_set if fulfills(feature_attribute, generalized_feature)}

            ############# Step 2 - > Looping over individual specializations i.e. (x, c(x)) ∈ D - Candidate Elimination Algorithm ##########
            for specialized_hypothesis_value in specialization_list:
                if specialized_hypothesis_value not in specialization_set:
                    continue
                if not fulfills(feature_attribute, specialized_hypothesis_value): # s(x) != 1
                    # S = S \ {s} - removing the value
                    specialization_set.remove(specialized_hypothesis_value)

                    # Performing S+ = min_generalizations(s, x)
                    minimized_generalization_s = min_generalizations(specialized_hypothesis_value, feature_attribute)

                    # Performing S = S ∪ {s} , i.e. updating only the generalizations
                    specialization_set.update([specialized_hyp for specialized_hyp in minimized_generalization_s if any([more_general(indexed_hyp, specialized_hyp)
                                                                        for indexed_hyp in generalization_set])])

                    # Performing S = S \ {s}, i.e. removing hypotheses less specific than any other in S
                    specialization_set.difference_update([specialized_hyp for specialized_hyp in specialization_set if
                                                          any([more_general(specialized_hyp, specialized_hyp_g) #any function used to fetch specific value
                                                               for specialized_hyp_g in specialization_set if specialized_hyp != specialized_hyp_g])])
        # when the indexed_feature is a negative example.
        elif not classes_list[index_of_feature]:
            generalization_list = list(generalization_set)
            specialization_set = {specialized_feature for specialized_feature in specialization_set if not fulfills(feature_attribute, specialized_feature)}
            ############# Step 2 - > Looping over individual generalizations i.e. g ∈ G ##########
            for generalized_hypothesis_value in generalization_list:

                if generalized_hypothesis_value not in generalization_set:
                    continue
                if fulfills(feature_attribute, generalized_hypothesis_value): #g(x) != 0

                    # G = G \ {g} - removing the value
                    generalization_set.remove(generalized_hypothesis_value)

                    # Performing G− = min_specializations(g, x)
                    minimized_specialization_g = min_specializations(generalized_hypothesis_value, domains, feature_attribute)

                    # Performing G = G ∪ {g} , i.e. updating only the specializations
                    generalization_set.update([specialized_hyp for specialized_hyp in minimized_specialization_g if any([more_general(specialized_hyp, specialized_hyp_s)
                                                        for specialized_hyp_s in specialization_set])])

                    # Performing G = G \ {g}, i.e. removing hypotheses less general than any other in G
                    generalization_set.difference_update([specialized_hyp for specialized_hyp in generalization_set if
                                         any([more_general(specialized_hyp_g, specialized_hyp) #any function used to fetch specific value
                                              for specialized_hyp_g in generalization_set if specialized_hyp != specialized_hyp_g])])

    ########### Step 3 - return(G, S) #########
    return [specialization_set,generalization_set]


########################################################################
# Tests
import os

dataset_file_name = os.path.join(os.path.dirname(__file__), 'dataset-web-pages-concept-learning.json')


def test_that_dataset_file_is_here():
    assert os.path.isfile(dataset_file_name), \
        "Please put the given dataset file next to this script!"


def test_more_general_relation():
    h1 = ("sunny", "warm", "normal", "strong")
    h2 = ("?", "warm", "?", "strong")
    h3 = ("sunny", "?", "normal", "strong")
    h4 = ("sunny", "?", "normal", "?")

    for h in [h1, h2, h3, h4, s_0(1), s_0(10), g_0(17)]:
        assert more_general(h, h), \
            f"h >=_g h should always be True, including for h={h}"

    assert not more_general(h1, g_0(len(h1))), \
        "no other hypothesis should be more general than g_0"
    assert more_general(g_0(len(h1)), h1), \
        "g_0 should be more general than any other hypothesis"

    assert more_general(h1, s_0(len(h1))), \
        "no hypothesis should be less general than s_0"
    assert not more_general(s_0(len(h1)), h1), \
        "s_0 should not be more general than any other hypothesis"

    for h_a, h_b in [(h2, h3), (h3, h2), (h2, h4), (h4, h2)]:
        assert not more_general(h_a, h_b), \
            "more_general should never return True for pairs of hypotheses " \
            f"like {h_a} and {h_b} " \
            "that are not descendants of each other in hypothesis space."

    for h_a, h_b in [(h2, h1), (h3, h1), (h4, h1), (h4, h3)]:
        assert more_general(h_a, h_b), \
            f"{h_a} is strictly more general than {h_b} but more_general returned False"
        assert not more_general(h_b, h_a), \
            f"{h_b} is strictly less general than {h_a} but more_general returned True"


class Examples:
    h0 = s_0(6)
    h1 = ('sunny', 'warm', 'normal', 'strong', 'warm', 'same')
    h2 = ('sunny', 'warm', '?', 'strong', 'warm', 'same')
    h3 = ('sunny', 'warm', '?', 'strong', '?', '?')
    h4 = ('sunny', 'warm', '?', 'strong', '?', '?')
    h5 = ('sunny', '?', '?', '?', '?', '?')
    h6 = ('cloudy', '?', '?', '?', '?', '?')
    h7 = ('?', 'warm', '?', '?', '?', '?')
    h8 = ('?', '?', 'normal', '?', '?', '?')
    h9 = ('?', '?', '?', 'light', '?', '?')
    h10 = ('?', '?', '?', '?', 'cool', '?')
    h11 = ('?', '?', '?', '?', '?', 'change')
    h12 = g_0(6)

    x1 = ('sunny', 'warm', 'normal', 'strong', 'warm', 'same')
    x2 = ('sunny', 'warm', 'high', 'strong', 'warm', 'same')
    x3 = ('rainy', 'cold', 'high', 'strong', 'warm', 'same')
    x4 = ('sunny', 'warm', 'high', 'strong', 'cool', 'change')
    x5 = ('cloudy', 'warm', 'normal', 'light', 'cool', 'change')

    domains = [
        ['sunny', 'rainy', 'cloudy'], ['warm', 'cold'], ['normal', 'high'],
        ['strong', 'light'], ['warm', 'cool'], ['same', 'change']
    ]


def test_generalizations():
    X = Examples

    for h_before, example, h_after in [
        (X.h0, X.x1, [X.h1]), (X.h1, X.x2, [X.h2]), (X.h2, X.x4, [X.h4])]:
        assert set(min_generalizations(h_before, example)) == set(h_after), \
            f"Hypothesis {h_before} should generalize to {h_after} given example {example}"


def test_specializations():
    X = Examples

    assert set(min_specializations(X.h12, X.domains, X.x3)) == set([
        X.h5, X.h6, X.h7, X.h8, X.h9, X.h10, X.h11
    ]), \
        f"min_specializations should return all minimal specializations"


def test_find_domains():
    X = Examples
    result = find_domains([X.x1, X.x2, X.x3, X.x4, X.x5])
    assert len(result) == len(X.x1), \
        "The number of attribute domains returned by find_domains should be the " \
        "same as the number of attributes in the given examples."
    assert [set(d) for d in result] == [set(d) for d in X.domains], \
        "find_domains should return all possible values for each feature."


def test_candidate_elimination():
    X = Examples
    S, G = candidate_elimination(
        [X.x1, X.x2, X.x3, X.x4],
        [True, True, False, True],
        X.domains)

    assert S == set([('sunny', 'warm', '?', 'strong', '?', '?')]), \
        "candidate_elimination should return the same set S "
    assert G == set([
        ('sunny', '?', '?', '?', '?', '?'), ('?', 'warm', '?', '?', '?', '?')
    ]), \
        "candidate_elimination should return the same set G "


########################################################################
# Main program for running againts the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

   
    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running your implementation on the training dataset")
    df = pd.read_json(dataset_file_name)
    examples = [tuple(x[:-1]) for x in df.values]
    target_concept = 'is_sell'
    classes = df[target_concept].values
    domains = find_domains(examples)
    S, G = candidate_elimination(examples, classes, domains)
    print(
        f"Your implementation found the sets S={S} and G={G} for the target concept {target_concept} and attributes {list(df.columns[:-1])}")
        
############################################################################################################################################################################        

########################################################################
#                Linear models for web site function detection.
########################################################################


def load_feature_vectors(filename: str) -> np.array:
    """Load the feature vectors from the dataset in the given file and return
    them as a numpy array with shape (number-of-examples, number-of-features).

    Convert feature vectors to expanded form in the process.
    """
    
    # loading dataset
    load_dataset = pd.read_json(filename)
    # performing removal of page_id, from required features
    length_of_features_classifications = int(len(load_dataset.columns) - 1)
    dataset_values = load_dataset.iloc[:, 0:length_of_features_classifications]
    feature_values = dataset_values.iloc[:, 1:length_of_features_classifications]
    # creating feature vectors, extended i.e. x = (1, x1, . . . , xp)
    extended_features_values = np.ones(dataset_values.shape)
    feature_values_num = np.array(feature_values, dtype=object)

    extended_features_values[:, 1:] = feature_values_num[:, :]

    return extended_features_values

def load_class_values(filename: str) -> np.array:
    """Load the class values from the dataset in the given file and return
    them as a one-dimensional numpy array"""
    
    # loading dataset
    load_dataset = pd.read_json(filename)
    # slicing 'sell' class
    remove_class_vector = load_dataset.iloc[:, -1:]

    class_values = np.array(remove_class_vector, np.dtype(object))

    return class_values

def linear_model(w: np.array, x: np.array) -> float:
    """Return the prediction of a linear regression model with parameter vector
    `w` on example `x`."""
    

    feature_weights = w
    feature_examples = x
    # performing wTx
    prediction_linear_model = np.dot(feature_weights, feature_examples)

    return prediction_linear_model

def logistic_model(w: np.array, x: np.array) -> float:
    """Return the prediction of a logistic regression model with parameter
    vector `w` on example `x`."""
    

    feature_weights = w
    feature_examples = x

    # performing wTx
    cross_weights_examples = np.dot(feature_weights, feature_examples)
    # performing sigmoid function
    prediction_linear_model = 1/(1 + np.exp(-(cross_weights_examples)))

    return prediction_linear_model

def initialize_random_weights(p: int) -> np.array:
    """Generate a pseudorandom weight vector of dimension p.

    The returned array be a column vector with p elements, i.e., it should have
    shape (1, p).

    If the parameter random_seed is not None, it should result in calls with the
    same seed having the same outcome.
    """
    
    no_of_elements = p

    random_weights = np.random.rand(1, no_of_elements)

    return random_weights

def train_linear_regression_with_lms(xs: np.array, cs: np.array, eta: float, t_max: int) -> np.array:
    """Fit a linear regression model using the Least Mean Squares algorithm and
    return the learned weights as a numpy array.

    Arguments:
    - `xs`: feature vectors in the training dataset as a two-dimensional numpy array with shape (n, p+1)
    - `cs`: class values c(x) for every element in `xs` as a one-dimensional numpy array with length n
    - `eta`: the learning rate as a float value
    - `t_max: the number of iterations to run the algorithm for

    Returns:
    - the learned weights as a column vector, i.e. a two-dimensional numpy array with shape (1, p)
    """
    
    feature_values = xs
    no_of_features = feature_values.shape[1]
    random_weights = initialize_random_weights(no_of_features)
    no_of_iterations = t_max
    class_values = cs
    learning_value = eta

    iteration = 0
    while iteration < no_of_iterations:
            random_feature_index = np.random.randint(feature_values.shape[0])
            prediction_y = linear_model(random_weights, feature_values[random_feature_index])
            sigma = class_values[random_feature_index] - prediction_y
            diff_w = np.multiply(feature_values[random_feature_index], learning_value * sigma)
            random_weights[0] = random_weights[0] + diff_w
            iteration += 1

    return random_weights

def train_logistic_regression_with_bgd(xs: np.array, cs: np.array, eta: float, t_max: int) -> np.array:
    """Fit a logistic regression model using the Batch Gradient Descent algorithm and
    return the learned weights as a numpy array.

    Arguments:
    - `xs`: feature vectors in the training dataset as a two-dimensional numpy array with shape (n, p+1)
    - `cs`: class values c(x) for every element in `xs` as a one-dimensional numpy array with length n
    - `eta`: the learning rate as a float value
    - `t_max: the number of iterations to run the algorithm for

    Returns:
    - the learned weights as a column vector, i.e. a two-dimensional numpy array with shape (1, p)
    """
    
    feature_values = xs
    no_of_features = feature_values.shape[1]
    random_weights = initialize_random_weights(no_of_features)
    no_of_iterations = t_max
    class_values = cs
    learning_value = eta

    iteration = 0
    while iteration < no_of_iterations:
        for indexed_feature in range(no_of_features):
            prediction_y = logistic_model(random_weights, feature_values[indexed_feature])
            sigma = class_values[indexed_feature] - prediction_y
            diff_w = np.multiply(feature_values[indexed_feature], learning_value * sigma)
            random_weights[0] = random_weights[0] + diff_w
        iteration += 1

    return random_weights

def misclassification_rate(truth: np.array, predictions: np.array) -> float:
    """Given two one-dimensional numpy arrays with classifications and model
    predictions, compute and return the misclassification rate"""
    
    size_of_classificaions = len(truth)
    count_of_misclassified_values = 0
    
    for classified_elements in range(size_of_classificaions):
        if truth[classified_elements] != predictions[classified_elements]:
            count_of_misclassified_values += 1
    
    misclassification_rate = count_of_misclassified_values / size_of_classificaions
    
    return misclassification_rate

def threshold( b : np.array, t: float = 0.5) -> np.array:
    """Return a thresholded predictions of a linear or logistic regression
    model. All values in the output should be either 0 or 1."""
    
    threshold_prediction = t
    predicted_model = b
    output_class_values = []

    for indexed_value in range(len(predicted_model)):
        if predicted_model[indexed_value] > threshold_prediction:
            output_class_values.append(1)
        elif predicted_model[indexed_value] < threshold_prediction:
            output_class_values.append(0)

    return output_class_values



def train_and_compare(dataset_filename: str):
    """Load the given dataset and split it into 80% training set and 20%
    hold-out set. Train a linear regression and a logistic regression model on
    the training set, and compare their misclassification rates on the hold-out set."""
    

    load_dataset = pd.read_json(dataset_filename)
    length_of_split_index = int(0.8 * len(load_dataset))
    feature_vectors = load_feature_vectors(dataset_filename)
    class_vectors = load_class_values(dataset_filename)
    training_set_values_features = feature_vectors[:length_of_split_index, :]
    training_set_values_class = class_vectors[:length_of_split_index, :]
    holding_set_values_features = feature_vectors[length_of_split_index:, :]
    holding_set_values_class = class_vectors[length_of_split_index:, :]
    learning_rate = 0.000001
    max_iterations_training = 10000
    max_iterations_hold = 100
    threshold_value = 0.5

    #Performing operations on 80% training set
    lms_model_predictions = train_linear_regression_with_lms(training_set_values_features, training_set_values_class, learning_rate, max_iterations_training)
    logistic_model_predictions = train_logistic_regression_with_bgd(training_set_values_features, training_set_values_class, learning_rate, max_iterations_training)

    prediction_y_weights_linear = []
    prediction_y_weights_logistic = []

    for feature_row in training_set_values_features:
        prediction_y_linear = linear_model(lms_model_predictions, feature_row)
        prediction_y_logistic = logistic_model(logistic_model_predictions, feature_row)
        prediction_y_weights_linear.append(prediction_y_linear)
        prediction_y_weights_logistic.append(prediction_y_logistic)

    correct_linear_predictions = threshold(prediction_y_weights_linear,threshold_value)
    correct_logistic_predictions = threshold(prediction_y_weights_logistic, threshold_value)

    error_training_set_linear = misclassification_rate(training_set_values_class, correct_linear_predictions)
    error_training_set_logistic = misclassification_rate(training_set_values_class, correct_logistic_predictions)

    # Performing operations on 20% holdout set
    lms_model_predictions_hold = train_linear_regression_with_lms(holding_set_values_features, holding_set_values_class, learning_rate, max_iterations_hold)
    logistic_model_predictions_hold = train_logistic_regression_with_bgd(holding_set_values_features, holding_set_values_class, learning_rate, max_iterations_hold)

    prediction_y_weights_linear_hold = []
    prediction_y_weights_logistic_hold = []

    for feature_row_hold in holding_set_values_features:
        prediction_y_linear_hold = linear_model(lms_model_predictions_hold, feature_row_hold)
        prediction_y_logistic_hold = logistic_model(logistic_model_predictions_hold, feature_row_hold)
        prediction_y_weights_linear_hold.append(prediction_y_linear_hold)
        prediction_y_weights_logistic_hold.append(prediction_y_logistic_hold)

    correct_linear_predictions_hold = threshold(prediction_y_weights_linear_hold, threshold_value)
    correct_logistic_predictions_hold = threshold(prediction_y_weights_logistic_hold, threshold_value)

    error_holding_set_linear = misclassification_rate(holding_set_values_class, correct_linear_predictions_hold)
    error_holding_set_logistic = misclassification_rate(holding_set_values_class, correct_logistic_predictions_hold)

    print(tabulate([['Err_Train', error_training_set_linear, error_training_set_logistic], ['Err_Test', error_holding_set_linear, error_holding_set_logistic]], headers=['Model', 'Linear', 'Logistic']))

########################################################################
# Tests
import os
from pytest import approx

train_data_file_name = os.path.join(os.path.dirname(__file__), 'dataset-web-pages-linear-models-train-handout.json')
test_data_file_name = os.path.join(os.path.dirname(__file__), 'dataset-web-pages-linear-models-test-handout.json')

def test_that_training_data_is_here():
    assert os.path.isfile(train_data_file_name), \
        "Please put the training dataset file next to this script!"


def test_load_feature_vectors():
    xs = load_feature_vectors(train_data_file_name)
    assert xs.shape == (610, 6), "Should return matrix with n rows and p+1 columns"
    assert np.all(xs[:, 0] == np.ones(xs.shape[0])), "Should return extended feature vectors"


def test_load_class_values():
    cs = load_class_values(train_data_file_name)
    assert cs.shape == (610, 1), "Should return vector with n rows"
    assert np.all((cs == 0) | (cs == 1)), "Class values should be 1 or 0"


def test_linear_model():
    x = np.array([1, 1, 2])
    assert linear_model(np.array([0, 0, 0]), x) == 0
    assert linear_model(np.array([1, 0, 0]), x) == 1
    assert linear_model(np.array([-1, -1, -1]), x) == -4


def test_logistic_model():
    x = np.array([1, 1, 2])
    assert logistic_model(np.array([0, 0, 0]), x) == approx(0.5)
    assert logistic_model(np.array([1e6, 1e6, 1e6]), x) == approx(1)
    assert logistic_model(np.array([-1e6, -1e6, -1e6]), x) == approx(0)
    assert logistic_model(np.array([1e6, -1e6, 0]), x) == approx(0.5)


def test_initialize_random_weights():
    first = initialize_random_weights(7)
    second = initialize_random_weights(7)

    assert np.all(first.shape == second.shape), \
        "Same p should result in same shape"
    assert len(first.shape) == 2 and first.shape[0] == 1 and first.shape[1] == 7, \
        "initialize_random_weights(p) should result in a p-length column vector"
    assert not np.all(first == second), \
        "Different calls should result in different random weights"


def test_lms():
    xs = np.array([
        [1, 0],
        [1, 1],
        [1, 2]
    ])
    cs = [0, 1, 2]
    assert train_linear_regression_with_lms(xs, cs, 0.1, 1000) == approx(np.array([[0.0, 1.0]]), abs=0.01)


def test_bgd():
    xs = np.array([
        [1, -1],
        [1, 2],
        [1, -2],
    ])
    cs = [0, 1, 0]

    w = train_logistic_regression_with_bgd(xs, cs, 0.1, 100)
    assert w @ [1, -1] < 0 and w @ [1, 2] > 0
    w = train_logistic_regression_with_bgd(-xs, cs, 0.1, 100)
    assert w @ [1, -1] > 0 and w @ [1, 2] < 0


def test_misclassification_rate():
    import numpy as np
    truth = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    preds = np.array([1, 1, 1, 1, 0, 0, 1, 1])
    assert misclassification_rate(truth, preds) == 0.5, \
        "Misclassification rate should be 0.5 with 4/8 wrong decisions"
    assert misclassification_rate(np.zeros(10), np.zeros(10)) == 0, \
        "Misclassification rate should be zero with 0/10 wrong decisions"
    assert misclassification_rate(np.zeros(10), np.ones(10)) == 1, \
        "Misclassification rate should be 1 with 10/10 wrong decisions"


def test_threshold():
    p = np.array([-1000, 0.4, 0, 100, 0.6, 1000])
    assert np.all(threshold(p, 0.5) == [0, 0, 0, 1, 1, 1])


########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running train_and_compare on the training dataset")
    train_and_compare(train_data_file_name)
    if os.path.isfile(test_data_file_name):
        print("Test data found. Running train_and_predict.")
        preds = train_and_predict(train_data_file_name, test_data_file_name)
                       
                              
############################################################################################################################################################################        
########################################################################
#                Web page classification with Naive Bayes
########################################################################



def class_priors(cs: np.ndarray) -> dict:
    """Compute the prior probabilities P(C=c) for all the distinct classes c in the given dataset.

    Args:
        cs (np.ndarray): one-dimensional array of values c(x) for all examples x from the dataset D

    Returns:
        dict: a dictionary mapping each distinct class to its prior probability
    """
    
    dict_x = {}
    length_of_Class = len(cs)

    for x in cs:
        if x in dict_x:
            dict_x[x] += 1
        else:
            dict_x[x] = 1

    for y in dict_x:
        dict_x[y] = dict_x[y] / length_of_Class

    return dict_x


def conditional_probabilities(xs: np.ndarray, cs: np.ndarray) -> dict:
    """Compute the conditional probabilities P(B_j = x_j | C = c) for all combinations of feature B_j, feature value x_j and class c found in the given dataset.

    Args:
        xs (np.ndarray): n-by-p array with n points of p attributes each
        cs (np.ndarray): one-dimensional n-element array with values c(x)

    Returns:
        dict: nested dictionary d with d[c][B_j][x_j] = P(B_j = x_j | C=c)
    """
    

    indexes_of_classes = {}

    for classifier_value in set(cs):
        for index_of_class, class_value in enumerate(cs):
            if (class_value == classifier_value):
                if classifier_value not in indexes_of_classes:
                    indexes_of_classes[classifier_value] = []
                indexes_of_classes[classifier_value].append(index_of_class)

    feature_count = xs.shape[1]

    conditional_probability = {}

    for index, classes in indexes_of_classes.items():
        no_of_features = {}
        prob_of_features = {}
        conditional_probability[index] = no_of_features
        for index_of_feature_value in range(feature_count):
            if index_of_feature_value not in no_of_features:
                no_of_features[index_of_feature_value] = {}
            if index_of_feature_value not in prob_of_features:
                prob_of_features[index_of_feature_value] = {}
            for index_of_feature_attribute in classes:
                feature_prob = xs[index_of_feature_attribute][index_of_feature_value]
                if feature_prob not in no_of_features[index_of_feature_value]:
                    no_of_features[index_of_feature_value][feature_prob] = 0
                no_of_features[index_of_feature_value][feature_prob] += 1
            no_of_unique_features = len(classes)
            for feature_value, count_of_feature in no_of_features[index_of_feature_value].items():
                feature_probability = count_of_feature / no_of_unique_features
                no_of_features[index_of_feature_value][feature_value] = feature_probability

    return conditional_probability


class NaiveBayesClassifier:

    def fit(self, xs: np.ndarray, cs: np.ndarray):
        """Fit a Naive Bayes model on the given dataset

        Args:
            xs (np.ndarray): n-by-p array of feature vectors
            cs (np.ndarray): n-element array of class values
        """
        
        self.prob_of_total_dataset = conditional_probabilities(xs, cs)

    # np.array('sunny cold high strong')

    def predict(self, x: np.ndarray) -> str:
        """Generate a prediction for the data point x

        Args:
            x (np.ndarray): a p-dimensional feature vector

        Returns:
            str: the most probable class for x
        """
        

        #sunny cold high strong
        #print(self.prob_of_total_dataset)


        maxprob = 0
        example = x
        for index in self.prob_of_total_dataset:
            prob_of_j = 1
            for feature_value in example:
                for sub_index in self.prob_of_total_dataset[index]:
                    if feature_value in self.prob_of_total_dataset[index][sub_index]:
                        prob_of_i = self.prob_of_total_dataset[index][sub_index][feature_value]
                prob_of_j *= prob_of_i
                if prob_of_j > maxprob:
                    maxprob = prob_of_j
                    probable_class = index

        return probable_class


########################################################################
# Tests
import os
from pytest import approx

train_data_file_name = os.path.join(os.path.dirname(__file__),
                                    'dataset-web-pages-statistical-learning-train-handout.json')
val_data_file_name = os.path.join(os.path.dirname(__file__), 'dataset-web-pages-statistical-learning-val-handout.json')
test_data_file_name = os.path.join(os.path.dirname(__file__),
                                   'dataset-web-pages-statistical-learning-test-handout.json')

def test_that_training_data_is_here():
    assert os.path.isfile(train_data_file_name), \
        "Please put the training dataset file next to this script!"


def test_that_validation_data_is_here():
    assert os.path.isfile(val_data_file_name), \
        "Please put the validation dataset file next to this script!"


def test_that_test_data_is_here():
    assert os.path.isfile(test_data_file_name), \
        "Please put the test dataset file next to this script!"


def test_class_priors():
    cs = np.array(list('abcababa'))
    priors = class_priors(cs)
    assert priors == dict(a=0.5, b=0.375, c=0.125)


def test_conditional_probabilities():
    cs = np.array(list('aabb'))
    xs = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [2, 0, 0],
        [2, 1, 0]
    ])

    p = conditional_probabilities(xs, cs)

    assert p['a'][0][1] == 0.5
    assert p['a'][0][0] == 0.5
    assert p['b'][0][2] == 1
    assert p['a'][1][0] == 0.5
    assert p['a'][1][1] == 0.5
    assert p['b'][1][0] == 0.5
    assert p['b'][1][1] == 0.5
    assert p['a'][2][1] == 0.5
    assert p['a'][2][0] == 0.5
    assert p['b'][2][0] == 1


### example 
xs_example = np.array([x.split() for x in """sunny hot high weak
sunny hot high strong
overcast hot high weak
rain mild high weak
rain cold normal weak
rain cold normal strong
overcast cold normal strong
sunny mild high weak
sunny cold normal weak
rain mild normal weak
sunny mild normal strong
overcast mild high strong
overcast hot normal weak
rain mild high strong""".split('\n')])

cs_example = np.array("no no yes yes yes no yes no yes yes yes yes yes no".split())


def test_classifier():
    clf = NaiveBayesClassifier()
    clf.fit(xs_example, cs_example)
    pred = clf.predict(np.array('sunny cold high strong'.split()))
    assert pred == 'no', 'should classify example correctly'


def test_classifier_unknown_value():
    clf = NaiveBayesClassifier()
    clf.fit(xs_example, cs_example)
    pred = clf.predict(np.array('sunny hot dry none'.split()))
    assert pred == 'no', 'should handle unknown feature values'


########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running train_and_predict.")
    preds = train_and_predict(train_data_file_name, val_data_file_name, test_data_file_name)

                             
############################################################################################################################################################################        

def feature_example_rawlength(html_document):
    """Example feature function that returns the number of characters in the
    HTML document
    """

    return len(html_document.get_raw_html())


def feature_example_number_of_dom_nodes(html_document):
    """Example feature function that returns the number of nodes in the DOM tree.
    """
    return len(list(sample_document.get_tree().document.query_selector_all('*')))


def feature_example_number_of_sentences(html_document):
    """Example feature function that returns the number of sentences in the
    document's text.
    """
    return len(html_document.get_textblob().sentences)


########################################################################
# Implement your own feature functions below. Assume that each of them receives
# an instance of the class HTMLDocument as input (see below)

def feature_return_number_of_tags(html_document):
    # TODO. rename this function to describe what it does
    # TODO. Place your code here.
    no_of_tags = len(html_document.get_tree().document.query_selector_all('*'))
    return no_of_tags


def feature_return_number_of_links(html_document):
    # TODO. rename this function to describe what it does
    # TODO. Place your code here.
    no_of_tags = len(html_document.get_tree().body.get_elements_by_tag_name('a'))
    return no_of_tags

def feature_return_fraction_of_links(html_document):
    linkElements = html_document.get_tree().body.get_elements_by_tag_name('a')
    linkElementsTextContent = 0;

    htmlTextContent = len(html_document.get_textblob())

    for k in range(len(linkElements)):
        linkTextCount = len(linkElements[k].text)
        linkElementsTextContent = linkTextCount + linkElementsTextContent

    fractionOfLinksToText = linkElementsTextContent/(htmlTextContent-linkElementsTextContent)

    return fractionOfLinksToText

#Feature can be used to detect page language, but the test case fails due to its return type as string
"""def feature_detect_page_language(html_document):
    htmlTextContent =  html_document.get_textblob().detect_language()
    print('#Feature 4 : The webpage is in the -> ', htmlTextContent , ' language')
    return htmlTextContent
"""

def feature_detect_nouns(html_document):
    htmlTextNouns = html_document.get_textblob().noun_phrases
    #print('#Feature 5 : The nouns available in the HTML are -> ',  htmlTextNouns)
    htmlTextNounsLength = len(htmlTextNouns)
    return htmlTextNounsLength

def feature_count_detect_occurences(html_document):
    # word example in a html can be helped to characterize a website as a one for tutorial/learning ones
    reference_word = html_document.get_textblob().word_counts['example']
    return reference_word

def feature_number_of_inner_childs(html_document):
    childNodesOfBody = html_document.get_tree().body.child_nodes
    numberOfParentNodes = len(childNodesOfBody)
    childNodesOfinnerBodyChilds = 0

    for x in range(numberOfParentNodes):
        childNodesOfinnerBodyChilds = len(childNodesOfBody[x].child_nodes) + childNodesOfinnerBodyChilds

    ratioOfChildNodes = childNodesOfinnerBodyChilds/numberOfParentNodes

    return ratioOfChildNodes

def feature_ratio_of_list_elements(html_document):
    #Considering the fact that we ave only 1 unordered list in the sample html

    listItem = html_document.get_tree().body.get_elements_by_tag_name('ul')
    childElementsOfList = len(listItem.get_elements_by_tag_name('li'))

    ratioOfChildListElements = childElementsOfList/len(listItem)

    return ratioOfChildListElements

def feature_check_if_advertisements(html_document):
    linkElements = html_document.get_tree().body.get_elements_by_tag_name('a')
    linkElementsWithAds = 0;
    adLinkTexts = "adservice"
    for x in range(len(linkElements)):
        linkAttr = linkElements[x].getattr('href')
        if adLinkTexts in linkAttr:
            linkElementsWithAds = 1 + linkElementsWithAds
        else:
            linkElementsWithAds = 0

    return linkElementsWithAds

def feature_check_if_external_source(html_document):
    scriptElements = html_document.get_tree().body.get_elements_by_tag_name('script')
    imgElements = html_document.get_tree().body.get_elements_by_tag_name('img')
    externalElements = 0;

    if len(scriptElements) != 0 :
        externalElements = 1 + externalElements
    else:
        externalElements = 0

    if len(imgElements) != 0 :
        externalElements = 1 + externalElements
    else:
        externalElements = 0

    return externalElements
    
##################################################################################################################################

########################################################################
#                    web site features.
########################################################################

class HTMLDocument(object):
    """A single HTML document with helper methods for parsing.
    """

    def __init__(self, html_string):
        self._tree = None
        self._plaintext = None
        self._blob = None
        self.html_string = html_string

    def get_raw_html(self):
        """Returns the raw HTML contents of the document as is."""
        return self.html_string

    def get_tree(self):
        """Returns the document's DOM tree.

        Refer to https://resiliparse.chatnoir.eu/en/stable/man/parse/html.html for usage information.
        """
        from resiliparse.parse.html import HTMLTree
        if not self._tree:
            self._tree = HTMLTree.parse(self.html_string)
        return self._tree

    def get_text_content(self):
        """Returns the document's visible plaintext content as a single string.
        """
        from resiliparse.parse.html import NodeType

        if not self._plaintext:
            self._plaintext = ' '.join(e.text.strip() for e in sample_document.get_tree().body
                                       if e.type == NodeType.TEXT
                                       and e.text.strip()
                                       and e.parent.tag not in ['script', 'style'])

        return self._plaintext

    def get_textblob(self):
        """Returns a TextBlob object representing the document's natural
        language content.

        Refer to https://textblob.readthedocs.io/ for usage information
        """
        from textblob import TextBlob
        if not self._blob:
            self._blob = TextBlob(self.get_text_content())
        return self._blob


########################################################################
# Tests

sample_document = HTMLDocument("""<html><head><title>Sample</title></head>
<body>
<div id="header"><h1>This is an example page</h1></div>
<div id="main">
<ul><li>one</li><li><a href="https://webis.de">two</a></li><li>three</li></ul>
</div>
</body>
</html>""")

def _find_feature_functions():
    import inspect
    this_module = inspect.getmodule(sample_document)
    ffs = []
    for fname, fn in inspect.getmembers(this_module, inspect.isfunction):
        if fname.startswith('feature_'):
            ffs.append(fn)
    return ffs


def test_that_htmldocument_returns_plaintext():
    assert sample_document.get_text_content() == "This is an example page one two three", \
        "The HTMLDocument class's HTML parsing is not working as expected!"


def test_that_htmldocument_returns_textblob():
    assert sample_document.get_textblob().noun_phrases == ["example page"], \
        "The HTMLDocument class's natural-language parsing is not working as expected!"

def test_that_there_are_feature_functions():
    fns = _find_feature_functions()
    num_example_fns = len(
        [fn for fn in fns if fn.__name__.startswith('feature_example')])
    assert (len(fns) - num_example_fns) >= 3, \
        "Please implement at least three feature functions!"


def test_that_feature_functions_have_been_renamed():
    fns = _find_feature_functions()
    fns = [fn for fn in fns if not fn.__name__.startswith('feature_example')]
    names = [fn.__name__ for fn in fns]
    assert all([not n.startswith('feature_myfeature') for n in names]), \
        "Please give your feature functions descriptive names!"


def test_that_feature_functions_have_numeric_results():
    import numbers
    fns = _find_feature_functions()
    for fn in fns:
        result = fn(sample_document)
        assert isinstance(result, numbers.Number), \
            f"Please make sure that all feature functions including {fn.__name__} return a number!"


if __name__ == "__main__":
    import pytest
    import sys
    import nltk
   # print(feature_example_rawlength(sample_document))
    print('#Feature 1 : The no. of tags in the html are -> ' , feature_return_number_of_tags(sample_document))
    print('#Feature 2 : The no. of links in the html are -> ' , feature_return_number_of_links(sample_document))
    print('#Feature 3 : The ratio of the text content that describes the link vs. other text content -> ', feature_return_fraction_of_links(sample_document))
   # feature_detect_page_language(sample_document)
    print('#Feature 4 : The no of nouns available in the HTML are -> ',  feature_detect_nouns(sample_document))
    print('#Feature 5 : The no. of times the word - example occurs -> ', feature_count_detect_occurences(sample_document), ' classifying the website which may contain tutorials/learnings/lessons')
   # feature_return_ratio_of_lists(sample_document)
   # feature_number_of_tokens(sample_document)
    print('#Feature 6 : The ratio of child nodes with respect to parent nodes in the body is -> ', feature_number_of_inner_childs(sample_document))
    print('#Feature 7 : The ratio of list elements with respect to parent list in the body is -> ',  feature_ratio_of_list_elements(sample_document))
    print('#Feature 8 : The number of ads present in the HTML are -> ', feature_check_if_advertisements(sample_document))
    print('#Feature 9 : The number of external sources present in the HTML are -> ', feature_check_if_external_source(sample_document))

    sys.exit(pytest.main(['--tb=short', __file__]))
    
##################################################################################################################################    
    
########################################################################
#            Web page classification with CART decision trees
########################################################################    


def most_common_class(cs: np.array):
    """Return the most common class value in the given array

    Arguments:
    - cs: a 1-dimensional array of length n, containing of the class values c(x) for
          every element x of a dataset D
    """
    # Counting the total count of classes
    count_of_classes = Counter(cs)

    #Fetching most common class value
    most_common_class_value = count_of_classes.most_common(1)[0][0]

    return most_common_class_value


def gini_impurity(cs: np.array) -> float:
    """Compute the Gini index for a set of examples represented by the list of
    class values

    Arguments:
    - cs: a 1-dimensional array of length n, containing of the class values c(x) for
          every element x of a dataset D
    """
    # Checking the unique elements present in the list
    no_of_unique_elements = len(Counter(list(cs)).keys())
    # Checking the length of elements
    total_length = len(cs);

    set_of_class = []
    count_of_elem = []
    probability_of_elements = 0

    for each_class in cs:
        count_of_x = 0;
        if (set_of_class.__contains__(each_class) == False):
            for neighbour_class in cs:
                if each_class == neighbour_class:
                    count_of_x += 1
            set_of_class.append(each_class)
        count_of_elem.append(count_of_x)

    for elem_cnt in count_of_elem:
        if elem_cnt != 0:
            prob_of_elem = elem_cnt / total_length
            probability_of_elements += prob_of_elem * prob_of_elem

    return abs(1 - probability_of_elements)


def gini_impurity_reduction(impurity_D: float, cs_l: np.array, cs_r: np.array) -> float:
    """Compute the Gini impurity reduction af a binary split.

    Arguments:
    - impurity_D: the Gini impurity of the entire document D set to be split
    - cs_l: an array with the class values of the examples in the left split
    - cs_r: an array with the class values of the examples in the right split
    """

    len_of_left_split = len(cs_l)
    len_of_right_split = len(cs_r)

    gini_impurity_for_cs_l = gini_impurity(cs_l)
    gini_impurity_for_cs_r = gini_impurity(cs_r)

    set_of_left_split = []
    count_of_left_split = []

    set_of_right_split = []
    count_of_right_split = []

    prob_of_elem_l = 0
    for each_class in cs_l:
        count_of_x = 0;
        if (set_of_left_split.__contains__(each_class) == False):
            for neighbour_class in cs_l:
                if each_class == neighbour_class:
                    count_of_x += 1
            set_of_left_split.append(each_class)
        count_of_left_split.append(count_of_x)

    for elem_cnt in count_of_left_split:
        if elem_cnt != 0:
            prob_of_elem_l += (elem_cnt / len_of_left_split) ** 2

    prob_of_elem_r = 0
    for each_class in cs_r:
        count_of_x = 0;
        if (set_of_right_split.__contains__(each_class) == False):
            for neighbour_class in cs_r:
                if each_class == neighbour_class:
                    count_of_x += 1
            set_of_right_split.append(each_class)
        count_of_right_split.append(count_of_x)

    for elem_cnt in count_of_right_split:
        if elem_cnt != 0:
            prob_of_elem_r += (elem_cnt / len_of_right_split) ** 2

    return impurity_D - ((gini_impurity_for_cs_r * prob_of_elem_r) + (gini_impurity_for_cs_l * prob_of_elem_l))


def possible_thresholds(xs: np.array, attribute: int) -> np.array:
    """Compute all possible thresholds for splitting the example set xs along
    the given attribute. Pick thresholds as the mid-point between all pairs of
    distinct, consecutive values in ascending order.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - attribute: an integer with 0 <= a < p, giving the attribute to be used for splitting xs
    """
    #Asserting sort on the values
    sorted_set = sorted(set(xs[:, attribute, 0]))
    possible_thresholds_values = []
    
    for index, v in enumerate(sorted_set):
        if index < len(sorted_set) - 1:
            threshold = sorted_set[index] + (sorted_set[index + 1] - sorted_set[index]) / 2
            possible_thresholds_values.append(threshold)

    return np.array(possible_thresholds_values)


def find_split_indexes(xs: np.array, attribute: int, threshold: float) -> Tuple[np.array, np.array]:
    """Split the given dataset using the provided attribute and threshold.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - attribute: an integer with 0 <= a < p, giving the attribute to be used for splitting xs
    - threshold: the threshold to be used for splitting (xs, cs) along the given attribute

    Returns:
    - left: a 1-dimensional integer array, length <= n
    - right: a 1-dimensional integer array, length <= n
    """
    # This function is provided for you.
    smaller = (xs[:, attribute, :] < threshold).flatten()
    bigger = ~smaller  # element-wise negation

    idx = np.arange(xs.shape[0])

    return idx[smaller], idx[bigger]


def find_best_split(xs: np.array, cs: np.array) -> Tuple[int, float]:
    """
    Find the best split point for the dataset (xs, cs) from among the given
    possible attribute indexes, as determined by the Gini index.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - cs: a 1-dimensional array of length n

    Returns:
    - the attribute index of the best split
    - the threshold value of the best split
    """
    # hints to start
    a_best = None
    threshold_best = None
    gini_reduction_best = 0
    gini_all = gini_impurity(cs)  # impurity of the example set D
    attributes = np.arange(xs.shape[1])  # attributes available for splitting

    for attribute_value in attributes:

        thresholds = possible_thresholds(xs, attribute_value)

        for threshold_value in thresholds:
            left, right = find_split_indexes(xs, attribute_value, threshold_value)
            if len(left) == 0 or len(right) == 0:
                continue
            current_gini_reduction_value = gini_impurity_reduction(gini_all, left, right)
            if current_gini_reduction_value >= gini_reduction_best:
                a_best = attribute_value
                threshold_best = threshold_value
                gini_reduction_best = current_gini_reduction_value

    return ([a_best, threshold_best])


class CARTNode:
    """A node in a CART decision tree
    """

    def __init__(self):
        self.left = None
        self.right = None
        self.attribute = None
        self.threshold = None
        self.label = None

    def set_label(self, label):
        self.label = label

    def set_split(self, attribute: int, threshold: float, left: 'CARTNode', right: 'CARTNode'):
        """Turn this node into an internal node splitting at the given attribute
        and threshold, with the given left and right subtrees.
        """
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right

    def classify(self, x: np.array):
        """Return the class value for the given example as predicted by this subtree

        Arguments:
        - x: an array of shape (p, 1)
        """
        # This method is provided for you
        if self.attribute is None:
            # this is a leaf node
            return self.label

        v = x[self.attribute]

        if v < self.threshold:
            return self.left.classify(x)
        else:
            return self.right.classify(x)

    def __repr__(self):
        return f"[label={self.label};{self.attribute}|{self.threshold};L={self.left};R={self.right}]"


def id3_cart(xs: np.array, cs: np.array, max_depth: int = None) -> CARTNode:
    
    node = CARTNode()
    label_of_node = most_common_class(cs)
    node.label = label_of_node
    gini_entire = gini_impurity(cs)

    if gini_entire == 0:
        return node
    else:
        atrribute, threshold = find_best_split(xs, cs)
        node.attribute = atrribute
        node.threshold = threshold

        left_node, right_node = find_split_indexes(xs, node.attribute, node.threshold)
        
        left_node_features = np.array([xs[x] for x in left_node])
        left_node_classifiers = np.array([cs[x] for x in left_node])
        node.left = id3_cart(left_node_features, left_node_classifiers)

        right_node_features = np.array([xs[x] for x in right_node])
        right_node_classifiers = np.array([cs[x] for x in right_node])
        node.right = id3_cart(right_node_features, right_node_classifiers)
        
    return node

class CARTModel:
    """Trivial model interface class for the CART decision tree.
    """

    def __init__(self, max_depth=None):
        self._t = None  # root of the decision tree
        self._max_depth = max_depth

    def fit(self, xs: np.array, cs: np.array):
        self._t = id3_cart(xs, cs, self._max_depth)

    def predict(self, x):
        return self._t.classify(x)


########################################################################
# Tests
import os
from pytest import approx

train_data_file_name = os.path.join(os.path.dirname(__file__), 'dataset-web-pages-decision-trees-train-handout.json')
val_data_file_name = os.path.join(os.path.dirname(__file__), 'dataset-web-pages-decision-trees-val-handout.json')
test_data_file_name = os.path.join(os.path.dirname(__file__), 'dataset-web-pages-decision-trees-test-handout.json')


def test_that_training_data_is_here():
    assert os.path.isfile(train_data_file_name), \
        "Please put the training dataset file next to this script!"


def test_that_validation_data_is_here():
    assert os.path.isfile(val_data_file_name), \
        "Please put the validation dataset file next to this script!"


def test_that_test_data_is_here():
    assert os.path.isfile(test_data_file_name), \
        "Please put the test dataset file next to this script!"


def test_most_common_class():
    cs = np.array(['red', 'green', 'green', 'blue', 'green'])
    assert most_common_class(cs) == 'green', \
        "Identify the correct most common class"


def test_gini_impurity():
    # should work with two classes
    cs = np.array(['a', 'a', 'b', 'a'])
    assert gini_impurity(cs) == approx(2 * 0.75 * 0.25), \
        "Compute the correct Gini index for a two-class dataset"

    # should also work with more classes
    cs = np.array(['a', 'b', 'c', 'b', 'a'])
    assert gini_impurity(cs) == approx(1 - (0.4 ** 2 + 0.4 ** 2 + 0.2 ** 2)), \
        "Compute the correct Gini index for a three-class dataset"


def test_gini_impurity_reduction():
    # cs = np.array(['a', 'a', 'b', 'a'])
    i_D = 0.375

    assert gini_impurity_reduction(i_D, np.array(['a', 'a']), np.array(['b', 'a'])) == approx(0.125), \
        "Compute the correct gini reduction for the first test split"

    assert gini_impurity_reduction(i_D, np.array(['a', 'a', 'a']), np.array(['b'])) == approx(0.375), \
        "Compute the correct gini reduction for the second test split"


def test_possible_thresholds():
    xs = np.array([
        [[1], [0]],
        [[0.5], [1]],
        [[0], [0]],
        [[1], [1]],
    ])

    # first attribute allows two possible split points
    assert possible_thresholds(xs, 0) == approx(np.array([0.25, 0.75])), \
        "Find all possible thresholds for the first attribute."

    # second attribute only one
    assert possible_thresholds(xs, 1) == approx(np.array([0.5])), \
        "Find all possible thresholds for the second attribute"


def test_find_split_indexes():
    xs = np.array([
        [[1], [0]],
        [[0.5], [1]],
        [[0], [0]],
        [[1], [1]],
    ])
    l, r = find_split_indexes(xs, 0, 0.75)
    assert all(l == np.array([1, 2])) and all(r == np.array([0, 3]))

    l, r = find_split_indexes(xs, 0, 0.25)
    assert all(l == np.array([2])) and all(r == np.array([0, 1, 3]))


def test_find_best_split():
    xs = np.array([
        [[1], [0]],
        [[0.5], [1]],
        [[0], [0]],
        [[1], [1]],
    ])
    cs = np.array(['a', 'a', 'c', 'a'])
    a, t = find_best_split(xs, cs)
    assert a == 0, "Choose the best attribute."
    assert t == 0.25, "Choose the best threshold."


def test_cart_model():
    xs = np.array([
        [[1], [0]],
        [[0.5], [1]],
        [[0], [0]],
        [[1], [1]],
    ])
    cs = np.array(['a', 'a', 'b', 'b'])
    tree = CARTModel()
    tree.fit(xs, cs)
    preds = [tree.predict(x) for x in xs]

    assert all(cs == preds), \
        "On a dataset without label noise, reach zero training error."


########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running train_and_predict.")
    preds = train_and_predict(train_data_file_name, val_data_file_name, test_data_file_name)
                   
##################################################################################################################################  
##################################################################################################################################  
##################################################################################################################################                     