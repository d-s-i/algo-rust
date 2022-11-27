use std::fmt::Debug;
use std::collections::HashMap;
use std::hash::Hash;

fn main() {
    let mut nums = vec![0, 10, 1444, 458, 5 , 468412];
    println!("Initial array: {:?}", nums);
    radix_sort(&mut nums);
    println!("returned arr: {:?}", nums);
}

fn radix_sort(arr: &mut Vec<usize>) {
    // get the amount of iteration required
    let largest_digits = most_digit(&arr);

    // inspect all digits up to the largest number
    for i in 0..largest_digits {

        // create a bucket for all digits
        let mut buckets: Vec<Vec<usize>> = vec![vec![]; 9];

        // populate the bucket for each digit of each number at index i
        populate_buckets(&mut buckets, arr.to_vec(), i);

        // repopulate the array with each num for each digit in order
        populate_arr_from_buckets(buckets, arr);
    }
}

fn quick_sort<T: PartialOrd + Copy + Debug>(arr: &mut Vec<T>, start: usize, end: usize) {

    println!("end: {}, start: {}, arr: {:?}", end, start, arr);
    if end < start {
        return;
    }

    let i = pivot_helper(arr, start, end);
    
    if i <= 0 {
        return;
    }
    
    quick_sort(arr, start, i - 1);
    quick_sort(arr, i + 1, end);
    
}

fn merge_sort<T: PartialOrd + Copy>(arr: Vec<T>) -> Vec<T> {
    if arr.len() <= 1 {
        return arr;
    }
    let mid = arr.len() / 2;
    let left = merge_sort(arr[..mid].to_vec());
    let right = merge_sort(arr[mid..].to_vec());

    merge_vecs(left, right)
}

fn insertion_sort(arr: &mut Vec<i32>) {
    for i in 1..arr.len() {
        let curr_val = arr[i];
        
        let mut j = isize::try_from(i - 1).ok().unwrap();
        loop {
            if j < 0 || arr[j as usize] <= curr_val {
                break;
            }

            arr[(j + 1) as usize] = arr[j as usize];

            j -= 1;
        }
        arr[(j + 1) as usize] = curr_val;
    }
}

fn selection_sort(arr: &mut Vec<i32>) {
    for i in 0..arr.len() {
        let mut index_lowest = i;
        for j in i..arr.len() {
            if arr[index_lowest] > arr[j] {
                index_lowest = j;
            }
        }
        swap_values(arr, i, index_lowest);
    }
}

/// arr must be sorted
fn bin_search(arr: Vec<i32>, num_to_find: i32) -> Option<usize> {
    let mut start = 0;
    let mut end = arr.len() - 1;

    let mut prev_i = 0;
    loop {
        let i = (start + end) / 2;
        if prev_i == i {
            return None;
        }
        let val = arr[i];
        if val == num_to_find {
            return Some(i);
        }

        if val > num_to_find {
            end = i - 1;
        } else {
            start = i + 1;
        }
        prev_i = i;
    }
}

/// Take a range and return the biggest sum possible this amount of number within the array
fn max_subarray_sum(arr: &Vec<i32>, window: usize) -> Option<i32> {
    if window > arr.len() {
        return None;
    }
    let mut initial_sum = 0;
    for i in 0..window {
        initial_sum += arr[i];
    };

    let mut temp_sum = initial_sum;
    let mut max = 0;
    let mut i = window;
    loop {
        if i == arr.len() {
            break;
        }
        temp_sum = temp_sum - arr[i - window] + arr[i];

        if temp_sum > max {
            max = temp_sum;
        }
        i += 1;
    }
    Some(max)
}

fn count_unique_values(list: &mut Vec<i32>) -> usize {
    let mut i = 0;
    for j in 0..list.len() {
        if list[i] != list[j] {
            i += 1;
            list[i] = list[j];
        }
    }
    i + 1
}

fn sum_zero(list: Vec<i32>) -> Option<(i32, i32)> {

    let mut left = 0;
    let mut right = list.len() - 1;
    loop {
        if left >= right {
            return None;
        }
        let sum = list[left] + list[right];

        if sum == 0 {
            return Some((list[left], list[right]));
        } else if sum > 0 {
            right -= 1;
        } else {
            left += 1;
        }
    }
}

/// Take an array `a`, map its values to square and tells if this arr is similar to `b`
fn squared_same(a: Vec<i32>, b: Vec<i32>) -> bool {
    let a_squared: Vec<i32> = a.iter().map(|val| val * val).collect();

    let mut freq_a: HashMap<i32, u32> = HashMap::new();
    for val_a in a_squared {
        let new_val = match freq_a.get_mut(&val_a) {
            Some(num) => *num + 1,
            None => 1
        };
        freq_a.insert(val_a, new_val);
    }

    let mut freq_b: HashMap<i32, u32> = HashMap::new();
    for val_b in b {
        let new_val = match freq_b.get_mut(&val_b) {
            Some(num) => *num + 1,
            None => 1
        };
        freq_b.insert(val_b, new_val);
    }

    for (key, freq_a_value) in freq_a.iter() {
        match freq_b.get(key) {
            Some(freq_b_value) => {
                if freq_a_value != freq_b_value {
                    return false;
                }
            },
            None => {
                return false;
            }
        }
    }
    
    true
}

fn swap_values<T: Copy>(arr: &mut Vec<T>, i: usize, j: usize) {
    let temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

fn merge_vecs<T: PartialOrd + Copy>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
    let mut arr = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    loop {
        if a[i] < b[j] {
            arr.push(a[i]);
            if i <= a.len() - 1 { 
                i += 1;
            }
        } else {
            arr.push(b[j]);
            if j <= b.len() - 1 { 
                j += 1;
            }
        }
        
        if i == a.len() {
            for remaining in b[j..b.len()].iter() {
                arr.push(*remaining);
            }
            break;
        }
        if j == b.len() {
            for remaining in a[i..a.len()].iter() {
                arr.push(*remaining);
            }
            break;
        }
    }

    arr
}

fn pivot_helper<T: Copy + PartialOrd>(arr: &mut Vec<T>, start: usize, end: usize) -> usize {
    let mut pivot_index = start;
    let pivot = arr[start];

    for i in start + 1..=end {
        if arr[i] < pivot {
            pivot_index += 1;
            swap_values(arr, i, pivot_index);
        }
    }
    swap_values(arr, start, pivot_index);
    pivot_index
}

fn get_digit(num: usize, digit_index: u32) -> usize {
    (num / 10_usize.pow(digit_index)) % 10
}

fn get_digit_count(num: usize) -> u8 {
    let n = i32::try_from(num).unwrap() as f32;
    (n.log10() as u8) + 1
}

fn most_digit(arr: &Vec<usize>) -> u8 {
    let mut max = 0;
    for val in arr.iter() {
        let digits = get_digit_count(*val);
        if digits > max {
            max = digits
        }
    }
    max
}


fn populate_buckets(buckets: &mut Vec<Vec<usize>>, nums: Vec<usize>, digit_index: u8) {
    for num in nums.iter() {
        let curr_digit = get_digit(*num, u32::try_from(digit_index).unwrap());
        buckets[curr_digit].push(*num);
    }
}

fn populate_arr_from_buckets(buckets: Vec<Vec<usize>>, arr: &mut Vec<usize>) {
    let mut index = 0;
    for bucket in buckets.iter() {
        if bucket.len() > 0 {
            for num in bucket {
                arr[index] = *num;
                index += 1;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    #[ignore]
    fn frequency_counter() {
        assert_eq!(squared_same(vec![1, 2, 3], vec![4, 1, 9]), true);
        assert_eq!(squared_same(vec![1, 2, 3], vec![4, 9]), false);
        assert_eq!(squared_same(vec![1, 2, 1], vec![4, 4, 1]), false);
    }

    #[test]
    #[ignore]
    fn multiple_pointer() {
        assert_eq!(sum_zero(vec![-3, -2, -1, 0, 1, 2, 3]).unwrap(), (-3, 3));
    }

    #[test]
    #[ignore]
    fn loop_test() {
        assert_eq!(count_unique_values(&mut vec![1, 1, 1, 2, 3, 3, 3, 3]), 3);
    }
    
    #[test]
    #[ignore]
    fn window_ptr() {
        let res = max_subarray_sum(&vec![1, 5, 99, 66, 1, 0, 0, 2, 4, 1, 2, 45], 2);
        assert_eq!(res.unwrap(), 165);
    }

    #[test]
    #[ignore]
    fn binary_search() {
        let res1 = bin_search(vec![1, 3, 4, 8, 9, 15, 20, 27, 29, 40, 45, 65, 88], 88);
        let res2 = bin_search(vec![1, 3, 4, 8, 9, 15, 20, 27, 29, 40, 45, 65, 88], 11);

        assert_eq!(res1, Some(12));
        assert_eq!(res2, None);
    }

    #[test]
    #[ignore]
    fn test_selection_sort() {
        let mut arr = vec![5, 8, 9, 7, 4, 14, 12];
        selection_sort(&mut arr);
        assert_eq!(arr, vec![4, 5, 7, 8, 9, 12, 14]);
    }

    #[test]
    #[ignore]
    fn test_insertion_sort() {
        let mut arr = vec![3, 5, 1, 2, 7, 9, 5, 4, 1, 2, 6, 8, 7, 44, 5, 65, 77];
        insertion_sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7, 8, 9, 44, 65, 77]);
    }

    #[test]
    #[ignore]
    fn merge_two_vecs() {
        let arr1 = vec![1, 2, 5, 8];
        let arr2 = vec![3, 4, 5, 6];
        let res = merge_vecs(arr1, arr2);
        assert_eq!(res, vec![1, 2, 3, 4, 5, 5, 6, 8]);
    }

    #[test]
    #[ignore]
    fn test_merge_sort() {
        let arr = vec![1, 5, 4, 7, 8, 9, 6, 5, 2, 6, 4];
        let res = merge_sort(arr);
        println!("{:?}", res);
    }

    #[test]
    #[ignore]
    fn test_pivot_helper() {
        let mut arr = vec![4, 8, 2, 1, 5, 7, 6, 3];
        let end = arr.len() - 1;
        let i = pivot_helper(&mut arr, 0, end);
        assert_eq!(arr, vec![3, 2, 1, 4, 5, 7, 6, 8]);
        assert_eq!(i, 3);
        // println!("Arr is not {:?}", arr);
        // println!("i: {}", i);
    }

    #[test]
    #[ignore]
    fn test_get_digit() {
        let num = 12345;
        let digit_0 = get_digit(num, 0);
        let digit_1 = get_digit(num, 1);
        let digit_2 = get_digit(num, 2);
        let digit_3 = get_digit(num, 3);

        assert_eq!(digit_0, 5);
        assert_eq!(digit_1, 4);
        assert_eq!(digit_2, 3);
        assert_eq!(digit_3, 2);
    }

    #[test]
    #[ignore]
    fn test_get_digit_count() {
        let a = get_digit_count(15684);
        assert_eq!(a, 5);

        let b = get_digit_count(0);
        assert_eq!(b, 1);
    }

    #[test]
    #[ignore]
    fn test_most_digits() {
        let nums = vec![0, 10, 1444, 458, 5 , 468412];
        let max = most_digit(&nums);

        assert_eq!(max, 6);
    }

    #[test]
    fn test_radix_sort() {
        let mut nums = vec![0, 10, 1444, 458, 5, 468412];
        radix_sort(&mut nums);

        assert_eq!(nums, [0, 5, 10, 458, 1444, 468412]);
    }
}