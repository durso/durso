<?php

namespace cache;
use app\model\file;
use library\utils;

class cache{
    private $controller;
    private $action;
    private $buffer;
    private $hasBuffer;
    private $extension = ".php";
    
    public function __construct($controller,$action){
        $this->controller = $controller;
        $this->action = $action;
        $this->hasBuffer = false;
    }
    
    public function init(){
        $this->file = __NAMESPACE__.DS.$this->controller.DS.$this->action.$this->extension;
        $this->hasBuffer = $this->isCached();
        return $this->hasBuffer;
    }
    public function run(){
        $this->buffer = ob_get_contents();
        if(!file::create($this->file, $this->buffer)){
            throw new \Exception("Could not create file: $this->file");
        }
        $this->hasBuffer = true;
    }
    private function isCached(){  
        $this->buffer = file::read($this->file);
        if($this->buffer !== false){
            return true;
        }
        return false;
    }
    
    private function isUpdated($date){
        $lastmodified = file::lastModified($this->file);
        $current = time();
        $period = strtotime($date,0);
        return $lastmodified + $period > $current;       
    }
    public function getCache(){
        if($this->hasBuffer){   
            utils::clearBuffer();
            echo $this->buffer;
        } else {
            utils::endFlush();
        }
    }
    public function setExtension($ext){
        $this->extension = $ext;
    }

}
