<?php

/**
 * Description of request
 *
 * @author durso
 */
namespace app;

class request {
    
    public static function isAjax(){
        return (!empty($_SERVER['HTTP_X_REQUESTED_WITH']) && strtolower($_SERVER['HTTP_X_REQUESTED_WITH']) == 'xmlhttprequest');
    }
    public static function url(){
        
    }
    
}