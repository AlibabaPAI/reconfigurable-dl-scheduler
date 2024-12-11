import pandas
def build_model(model_type,hidden_size,num_attention_heads,hidden_layers,atomic_bsz):
    # Hack, profile forward time in advance
    trace = pandas.read_csv("./simulator/profile_fwd/"+model_type+".csv")
    return trace.forward[0],trace.parameter_size[0]