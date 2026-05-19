<?php
/**
 * Plugin Name:  Merch Configurator
 * Description:  WooCommerce integration for the someoneWondered merch configurator.
 *               Reads custom cart params (merch_text, merch_shirt_color, merch_text_color, merch_size)
 *               and stores them as order item meta data.
 * Version:      1.0.0
 * Author:       Adam Fehse
 */

if (!defined('ABSPATH')) exit;

add_filter('woocommerce_add_cart_item_data', 'mcf_add_cart_item_data', 10, 2);
function mcf_add_cart_item_data($cart_item_data, $product_id) {
    if (!empty($_GET['merch_text'])) {
        $cart_item_data['merch_text'] = sanitize_text_field($_GET['merch_text']);
    }
    if (!empty($_GET['merch_shirt_color'])) {
        $cart_item_data['merch_shirt_color'] = sanitize_hex_color($_GET['merch_shirt_color']);
    }
    if (!empty($_GET['merch_text_color'])) {
        $cart_item_data['merch_text_color'] = sanitize_hex_color($_GET['merch_text_color']);
    }
    if (!empty($_GET['merch_size'])) {
        $cart_item_data['merch_size'] = strtoupper(sanitize_text_field($_GET['merch_size']));
    }
    return $cart_item_data;
}

add_filter('woocommerce_get_item_data', 'mcf_display_cart_item_data', 10, 2);
function mcf_display_cart_item_data($item_data, $cart_item) {
    $labels = [
        'merch_text'        => 'Custom Text',
        'merch_shirt_color' => 'Shirt Color',
        'merch_text_color'  => 'Text Color',
        'merch_size'        => 'Size',
    ];
    foreach ($labels as $key => $label) {
        if (!empty($cart_item[$key])) {
            $value = $cart_item[$key];
            if (in_array($key, ['merch_shirt_color', 'merch_text_color'])) {
                $value = '<span style="display:inline-block;width:1em;height:1em;border-radius:3px;background:' . esc_attr($value) . ';vertical-align:middle;margin-right:4px;border:1px solid #ccc;"></span> ' . esc_html($value);
            }
            $item_data[] = [
                'name'  => $label,
                'value' => $value,
            ];
        }
    }
    return $item_data;
}

add_action('woocommerce_checkout_create_order_line_item', 'mcf_save_order_item_meta', 10, 4);
function mcf_save_order_item_meta($item, $cart_item_key, $values, $order) {
    $fields = ['merch_text', 'merch_shirt_color', 'merch_text_color', 'merch_size'];
    foreach ($fields as $field) {
        if (!empty($values[$field])) {
            $item->add_meta_data($field, $values[$field]);
        }
    }
}
