import numpy as np

from crossview_geometry_dense_gt import direct_cycle, pair_summary, summarized_attempts


def test_descriptor_chain_is_not_a_direct_three_view_cycle():
    track = dict(conflict=False, n_views=3, observations={10: 4, 20: 7, 30: 9})
    edges = {(10, 20): {(4, 7)}, (20, 30): {(7, 9)}}
    assert not direct_cycle(track, edges)
    edges[(10, 30)] = {(4, 8)}
    assert not direct_cycle(track, edges)
    edges[(10, 30)] = {(4, 9)}
    assert direct_cycle(track, edges)


def test_conflicting_observations_never_form_an_accepted_cycle():
    track = dict(conflict=True, n_views=3, observations={10: 4, 20: 7, 30: 9})
    edges = {(10, 20): {(4, 7)}, (20, 30): {(7, 9)}, (10, 30): {(4, 9)}}
    assert not direct_cycle(track, edges)


def test_large_computable_error_is_separate_from_geometric_accuracy():
    rows = [dict(track_id=1, n_views=3, conflict=False, min_angle=1., reason='ok', error_px=100.),
            dict(track_id=1, n_views=3, conflict=False, min_angle=1., reason='outside_image', error_px=200.)]
    result = summarized_attempts(rows)
    assert result['computational_failure_rate'] == .5
    assert result['computed_inside_image_median_px'] == 100.
    assert result['all_attempts_under_4px_fraction'] == 0.
    assert result['valid_track_fraction_4px'] == 0.
    assert result['all_finite_error_p90_px'] > 100.


def test_low_support_does_not_dilute_supported_pair_statistics():
    rows = [dict(low_support=True, epi_median_px=.1, epi_p90_px=.2, n_matches=3),
            dict(low_support=False, epi_median_px=5., epi_p90_px=15., n_matches=80)]
    result = pair_summary(rows)
    assert result['n_pairs'] == 2
    assert result['supported_pairs'] == 1
    assert result['supported_median_pair_epi_px'] == 5.
    assert result['supported_median_pair_p90_px'] == 15.
    assert np.isclose(result['median_pair_epi_px'], 2.55)
