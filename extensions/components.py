from loko_extensions.model.components import Select, Dynamic, Input, Output, Component, save_extensions, Arg

imbalanced_data_description = """#### Imbalanced Data"""

target = Arg(name="target", label="Target Variable Name", type="text", value="target")
method = Select(name="method", label="Sampling Method", options=["undersampling", "oversampling", "SMOTE"],
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

args = [target, method, random_state, sampling_strategy_under, replacement, sampling_strategy_over, sampling_strategy_smote, k_neighbors]
input_list = [Input(id="balancing", label="Balancing", to="balancing", service="balance")]
output_list = [Output(id="balancing", label="Balancing")]
text_gen_component = Component(name="Imbalanced Data", inputs=input_list, outputs=output_list, args=args,
                               description=imbalanced_data_description, icon="RiScales3Fill")

if __name__ == '__main__':
    save_extensions([text_gen_component])
