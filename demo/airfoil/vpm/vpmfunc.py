def Get_LiftCoefficients(V,S,M):
    """ Calculates the lift coefficient of the wing

    This function computes the coefficient of lift for an airfoil as to
    be used in the vortex panel method "Get_PanelCoefficients"

    Args:
        V: The dimensionlesss velocity at each control point
        S: The dimensionless length of each of the control points
        M: The number of panels

    Returns:
        float: The coefficient of lift

    """

    gamma = 0
    for j in range(M):
        gamma = gamma + V[j]*S[j]

    cl = (2*gamma)

    return cl