"""Output of mobility prediction model for CBSA 14460. """

def predict_mode_probs():
    if trip_leg_miles <= 1.3960000276565552:
        if trip_leg_miles <= 0.7950000166893005:
            if hh_income <= 9.5:
                return [0.16, 0.04, 0.76, 0.05]
            else:  # if hh_income > 9.5
                return [0.09, 0.52, 0.41, 0.00]
        else:  # if trip_leg_miles > 0.7950000166893005
            if age <= 46.5:
                return [0.07, 0.80, 0.11, 0.05]
            else:  # if age > 46.5
                return [0.62, 0.00, 0.29, 0.14]
    else:  # if trip_leg_miles > 1.3960000276565552
        if age <= 68.5:
            if trip_leg_miles <= 16.79400062561035:
                return [0.29, 0.11, 0.05, 0.56]
            else:  # if trip_leg_miles > 16.79400062561035
                return [0.99, 0.00, 0.00, 0.10]
        else:  # if age > 68.5
            if age <= 74.5:
                return [0.54, 0.48, 0.00, 0.00]
            else:  # if age > 74.5
                return [1.05, 0.00, 0.00, 0.00]
