<?php
namespace app;


class bootstrap{
    private $controller = __NAMESPACE__."\\controller";
    private $action;
    private $query;
    private $error = false;

            
    public function __construct($get){
        $this->query = $get;
        
    }
    private function validate(){
        if(empty($this->query['controller'])){
            $this->controller .= "\\index";
        } else {
            if (preg_match("/([A-Za-z]+)/", $this->query['controller'])) {
                $this->controller .= "\\".$this->query['controller'];
            } else {
                $this->controller .= "\\apologize";
                $this->action = "index";
                return false;
            }
        }
        if(empty($this->query['action']) || preg_match("/([A-Za-z]+)/", $this->query['action']) === false){
            $this->action = "index";        
        } else {
            $this->action = $this->query['action'];
        }
        return true;
    }
        
    private function router(){
        //TODO
        return false;
    }     
    
    
    public function createController() {
        if(!$this->router()){
            $this->error = $this->validate();
        }
        if(!$this->error){
          return new $this->controller($this->action,$this->query,"Could not load the page");  
        }    
        return new $this->controller($this->action,$this->query);

    }
    
}

 
