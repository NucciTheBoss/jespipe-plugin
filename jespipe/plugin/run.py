def run(parameters=None, build=None, fit=None, predict=None, evaluate=None):
    """Function to launch the model functions in pipeline.
    
    Keyword arguments:
    params -- parameters collected by jespipe.plugins.start().
    build -- class used to build models.
    fit -- class used to fit models to datasets.
    predict -- class used to make predictions on test training sets.
    evaulatue -- class used to evaluate model performance."""
    global_error_template = "No value passed for {}. Please specify a value."

    if params is None:
        raise ValueError(global_error_template.format("params"))

    elif build is None:
        raise ValueError(global_error_template.format("build"))

    elif fit is None:
        raise ValueError(global_error_template.format("fit"))

    elif predict is None:
        raise ValueError(global_error_template.format("predict"))

    elif evaluate is None:
        raise ValueError(global_error_template.format("evaluate"))

    else:
        if params[0] == "train":
            build.build_model(params[1])

        elif params[0] == "attack":
            model = fit.model_fit(params[1])
            prediction = predict.model_predict(params[1], model)
            evaluate.model_evaluate(prediction)

        else:
            pass
