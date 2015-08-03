<?php
namespace app\model;
/**
 * File Class
 *
 * @author durso
 */
class file {
    public static function create($file, $buffer){
        $fp = self::open($file,"w");
        return self::write($fp,$buffer);
    }
    public static function append($file, $buffer){
        if(self::isReadable($file)){
            $fp = self::open($file,"a");
            return self::write($fp,$buffer);
        }
        return false;
    }
    public static function write($fp,$buffer){
        if($fp !== false){
            fwrite($fp,$buffer);
            fclose($fp);
            return true;
        } else{
            return false;
        }
    }
    public static function read($file){
        if(self::isReadable($file)){
            return file_get_contents($file);
        }
        return false;
    }
    public static function delete($file){
        return unlink($file);
    }
    public static function isReadable($file){
        return is_readable($file); 
    }
    public static function open($file,$mode){
        return fopen($file,$mode);  
    }
    public static function lastModified($file){
        return filemtime($file);
    }
}
