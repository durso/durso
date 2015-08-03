<?php


namespace library\dom\structures;
use library\dom\structures\components;
use library\dom\elements\components\button;



class buttonGroup extends components{
    public function __construct($class = 'btn-group'){
        parent::__construct("div");
        $this->root->addClass($class);
        $this->root->attr("role","group");
    }
 
    public function addButton($value){
        if($this->justified()){
            $div = new buttonGroup();
            $button = $div->addButton($value);
            $this->root->addComponent($div->save());
        } else {
            $button = new button($value);
            $button->attr("role","group");
            $this->root->addComponent($button);
        }
        $this->tracker($button);
        return $button;
    }
    public function size($classSize){
       $this->root->addClass($classSize);
    }
    public function nest(dropdown $dropdown){
       $dropdown->grouping();
       $this->root->addComponent($dropdown->save());
    }
    private function justified(){
       return $this->root->hasClass('btn-group-justified');
    }
    public function save(){
       return $this->root;
    }
}