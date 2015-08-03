<?php
namespace app;

class loader{
   
    public static function loadFile($file) {
        $fileName = '';
        $namespace = '';

        if (false !== ($lastNsPos = strripos($file, '\\'))) {
            $namespace = substr($file, 0, $lastNsPos);
            $file = substr($file, $lastNsPos + 1);
            $fileName = str_replace('\\', DS, $namespace) . DS;
        } 
        $fileName .= str_replace('_', DS, $file) . '.php';

        try{
            if (file_exists($fileName)) {
                require $fileName;
            } else {
                throw new \Exception("File $fileName does not exist.");
            }
        } catch (\Exception $e){
            echo $e->getMessage();
            error_log($e->getMessage().", file ".$e->getFile().", line ".$e->getLine());
            exit;
        }
    }
    public function register(){
        spl_autoload_register(array($this,"loadFile"));        
    }
    
    
}

 
