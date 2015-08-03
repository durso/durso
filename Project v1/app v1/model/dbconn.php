<?php

/**
 * Description of dbconn
 *
 * @author durso
 */
namespace app\model;
use library\utils;


class dbconn {
    private static $instance = null;
    
    public static function getInstance()  {   
        if (!(self::$instance instanceof \PDO)) { 
            try{
                self::$instance = new \PDO('mysql:host=localhost;dbname=docguide_brasil;charset=utf8', "root", "rd321678"); 
            } catch(\PDOException $e){
                utils::log($e->getMessage());
                return false;
            }
        }
        return self::$instance; 
    }
 }