from spike_handling import spike_io


def test_sort_by():
    list_to_sort = [7, 34, 1, 4, 1000, 20]
    list_to_sort_by = list(range(6))
    sorted_list, sort_by_list = spike_io.sort_by(list_to_sort, list_to_sort_by, descend=False)
    assert(sorted_list == list_to_sort)
    sorted_list, sort_by_list = spike_io.sort_by(list_to_sort, list_to_sort_by, descend=True)
    assert(sorted_list == [20, 1000, 4, 1, 34, 7])
