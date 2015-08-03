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
    public static function readCSV($file){
        return array_map('str_getcsv', file($file));
    }
    public static function opendir($dir){
        if(is_dir($dir)) {
            if ($dh = opendir($dir)) {
                
                return $dh;
            }
            
        }
        return false;
    }
    public static function scandir($dir,$filesOnly = true){
        $files = array();
        $dh = self::opendir($dir);
        if($dh){
            while (($file = readdir($dh)) !== false) {
                if($filesOnly){
                    if(filetype($dir.$file) == 'file'){
                        $files[] = $file;
                    }
                } else {
                    $files[] = $file;
                }
            }
            closedir($dh);
            return $files;
        }
        return false;
    }
    public static function getFiles($dir,$fileType){
        $files = self::scandir($dir);
        $list = array();
        if($files){
            foreach($files as $file){
                $ext = pathinfo($dir.$file, PATHINFO_EXTENSION);
                if($ext == $fileType){
                    $list[] = $file;
                }
            }
            return $list;
        }
        return false;
    }
}
