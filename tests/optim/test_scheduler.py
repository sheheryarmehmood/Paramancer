from paramancer.optim.scheduler import MomentumScheduler


def test_momentum_scheduler_uses_scalar_state_and_restarts():
    scheduler = MomentumScheduler()

    first = scheduler()
    second = scheduler()

    assert isinstance(scheduler.t, float)
    assert isinstance(first, float)
    assert isinstance(second, float)
    assert first == -1.0
    assert second == 0.0

    scheduler.restart()
    assert scheduler.t == 0.0
