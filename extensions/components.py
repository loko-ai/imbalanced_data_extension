from loko_extensions.model.components import Select, Dynamic, Input, Output, Component, save_extensions, Arg

imbalanced_data_description = """
### Description

Balancing a dataset makes training a model easier as it mitigates the risk of the model becoming biased towards a particular class. Imbalanced Data is the LOKO AI block designed to balance your data before passing it to the Predictor block. If you intend to link the Imbalanced Data component to a Predictor, note that you have to untoggle the stream button in the latter component settings.

This components it's based on the ImbalancedLearn python library, check their documentation if you want to know more https://imbalanced-learn.org/stable/references/index.html#api.
This component is based on the ImbalancedLearn python library. You can check their documentation to learn more: [Imbalanced Learn API](https://imbalanced-learn.org/stable/references/index.html#api).


### How to use it


You can balance your dataset by passing the CSVReader content to the Grouper component (setting a high number in the "Group size") and then linking its output to the "Balancing" input of the Imbalanced Data component.

### Settings


This block includes several fields to be set:

- **Target Variable Name**: By default, the value is "target," representing the name of the variable to consider as the target for sampling.
- **Sampling Method**: Choose among the 4 available sampling methods - undersampling, oversampling, SMOTE (Synthetic Minority Oversampling Technique), and SMOTEN (Synthetic Minority Over-sampling Technique for Nominal). The default method used is undersampling.
- **Random State**: You can set the random state to use.


Based on the chosen sampling strategy, you may find other fields:

- **Sampling Strategy**: This field is present independently of the chosen sampling method. It represents the resampling strategy to adopt. For example, choosing "minority" will imply resampling only the minority class, "not minority" will resample all classes except the minority, and so on.
- **k_neighbors**: This value can be set only for the synthetic-based techniques (i.e., "SMOTE," "SMOTEN"), and it represents the number of nearest neighbors used to define the neighborhood of samples used to generate the synthetic samples.
- **Sample with replacement**: This field is available to be set only for the "undersampling" method. If toggled, the sample will be with replacement of the extracted sample; otherwise, without.



"""

target = Arg(name="target", label="Target Variable Name", type="text", value="target")
method = Select(name="method", label="Sampling Method", options=["undersampling", "oversampling", "SMOTE", "SMOTEN"],
                value="undersampling")
random_state = Arg(name="random_state", label="Random State", type="number", value=123)
sampling_strategy_under = Dynamic(name="sampling_strategy", label="Sampling Strategy", dynamicType="select",
                                  options=["auto", "majority", "not majority", "not minority", "all"], value="auto", parent="method",
                                  condition='{parent}=="undersampling"')
replacement = Dynamic(name="replacement", label="Sample with raplacement", value=False, dynamicType="boolean",
                      parent="method", condition='{parent}=="undersampling"')

sampling_strategy_over = Dynamic(name="sampling_strategy", label="Sampling Strategy", dynamicType="select",
                                 options=['all', 'auto', 'not minority', 'not majority', 'minority'], value="auto", parent="method",
                                 condition='{parent}=="oversampling"')

sampling_strategy_smote = Dynamic(name="sampling_strategy", label="Sampling Strategy", dynamicType="select",
                                  options=['all', 'auto', 'not minority', 'not majority', 'minority'], value="auto", parent="method",
                                  condition='{parent}=="SMOTE"')

k_neighbors = Dynamic(name="k_neighbors", label="K neighbors", dynamicType="number", parent="method", value=5,
                      condition='{parent}=="SMOTE"')

sampling_strategy_smoten = Dynamic(name="sampling_strategy", label="Sampling Strategy", dynamicType="select",
                                  options=['all', 'auto', 'not minority', 'not majority', 'minority'], value="auto", parent="method",
                                  condition='{parent}=="SMOTEN"')

k_neighbors_smoten = Dynamic(name="k_neighbors", label="K neighbors", dynamicType="number", parent="method", value=5,
                      condition='{parent}=="SMOTEN"')

args = [target, method, random_state, sampling_strategy_under, replacement, sampling_strategy_over, sampling_strategy_smote, k_neighbors, sampling_strategy_smoten, k_neighbors_smoten]
input_list = [Input(id="balancing", label="Balancing", to="balancing", service="balance")]
output_list = [Output(id="balancing", label="Balancing")]
text_gen_component = Component(name="Imbalanced Data", inputs=input_list, outputs=output_list, args=args,
                               description=imbalanced_data_description, icon="RiScales3Fill")

if __name__ == '__main__':
    save_extensions([text_gen_component])
