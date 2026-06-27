def mape_score(model, x_test, y_test, transformer=None, verbose=True):
    """
    This function will produce mean absolute precision error (MAPE) results.
    If you transformed the y-variable before making predictions, make sure
    to use the transformer argument so that your results are evaluated fairly.


    PARAMETERS
    ----------
    model       | a fitted model object
    x_test      | x-data testing set
    y_test      | y-data testing set
    transformer | Use log if the y-variable was log transformed. Otherwise use an inverse transformer.
    verbose     | Whether to print MAPE results


    EXAMPLE USAGE
    -------------
    # no y-variable transformation while predicting
    mape_score(model  = my_model,
               x_test = x_test,
               y_test = y_test)


    # log_y transformation while predicting
    mape_score(model       = my_model,
               x_test      = x_test,
               y_test      = y_test,
               transformer = "log")


    # other y-transformation while predicting
    mape_score(model       = my_model,
               x_test      = x_test,
               y_test      = y_test,
               transformer = transformer)
    """
    preds  = model.predict(x_test, verbose=0).flatten()
    y_true = np.array(y_test).flatten()

    # inversing transformers
    if transformer:
        if transformer == 'log':
            y_true = np.exp(y_true)
            preds  = np.exp(preds)
        else:
            y_true = transformer.inverse_transform(y_true.reshape(-1, 1)).ravel()
            preds  = transformer.inverse_transform(preds.reshape(-1, 1)).ravel()

    indiv_mape     = np.abs((y_true - preds) / (y_true + 1e-8)) * 100
    overall_mape   = float(np.mean(indiv_mape))
    pct_within_20  = float(np.mean(indiv_mape < 20) * 100)

    # outputs
    if verbose == True:
        print(f"""
Overall MAPE: {overall_mape}

* Lower MAPE values are better.
            """)

    # returning results
    return {
        "overall_mape"  : round(overall_mape,  4)
