# Helper function to convert a coordinate pair of number:letter to cartesian number:number pair
def mb_coord_to_cartesian(mb_coord):
    column = mb_coord[0]
    row = mb_coord[1:]
    # Here we flip the row value so instead of being [1, 18] it is [18, 1] and convert the column coordinate from a
    # letter to a number [1, 11]
    return ord(column.upper()) - 65, - int(row) + 18


# An improvement here would be to figure out how much difference 25 degrees vs 40 degrees makes on the grade, if we can
# normalize the angle to adjust the grade we can double our dataset, it might also be possible to represent climbs of
# different angles with different values which would indicate a difference to the model
def moonboard_route_filter(route):
    return (
        route["Method"] == "Feet follow hands"
        and route["MoonBoardHoldSetup"] == "MoonBoard Masters 2017"
        and route.get("MoonboardConfiguration", "") == "40° MoonBoard"
    )
