class Vehicle(object):
    pass


class FourWheelVehicles(Vehicle):
    N_WHEELS = 4


class Car(Vehicle):
    N_CARS_CREATED = 0

    def __init__(self, color):
        speed = 10
        self.color = color
        self.n_wheels = FourWheelVehicles.N_WHEELS
        Car.N_CARS_CREATED += 1
        print(Car.N_CARS_CREATED)

