# Elevator Simulation

A tick-based elevator simulator coded in Java. It is built around a finite state machine with 7 states and a call prioritization scheduler. The repository also includes JavaFX visualization of the elevator's actions and 89 comprehensive JUnit 5 tests that validate 70+ scenarios.

*This was the capstone project for Advanced Data Structures and Embedded Systems.*

## Features

- State machine: `STOP → MVTOFLR → OPENDR → OFFLD → BOARD → CLOSEDR → MV1FLR`, decided every tick.
- Smart scheduling: chooses calls by demand above and below the car and by distance and reverses direction only when nothing is left ahead, just like a real elevator.
- Realistic passengers: Incorporates capacity limits and skips. Passengers give up when they've had to wait for too long. Certain riders hold closing doors open to let others in while others don't.
- Configurable timing: floor travel, door speed and boarding rate are all set in ticks.
- Analytics: a full event log plus per-passenger wait and trip times are exported to a csv file.

## Structure

```
src/building/      Building (state machine engine), Elevator, CallManager, Floor
src/passengers/    Passenger group model
src/genericqueue/  Generic bounded queue
src/               Controller, JavaFX GUI, JUnit test suites
test_data/         Scenario CSVs for testing
```

The code is split MVC-style. The JavaFX GUI is the view, `ElevatorSimController` runs the tick loop, and `Building` is the model.

## Running

It requires Java, JavaFX, JUnit 5 and `library/cmpElevator.jar` on the classpath.

1. Set the building parameters in `ElevatorSimConfig.csv` (floors, capacity, timing, passenger file).
2. Launch `ElevatorSimulation`, then use Run or Step.
3. Run the `BuildingFSM*Test` suites. They check the simulator's logs against reference logs.
